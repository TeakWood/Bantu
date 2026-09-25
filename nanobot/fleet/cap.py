"""The memory cap: when a fleet instance's process tree has used too much.

:mod:`nanobot.fleet.memory` answers "how many resident bytes does this process
group hold?" and stops there. This module is the policy built on that number —
the threshold comparison, the timing rule that makes the threshold meaningful,
and the record of which instances have already been caught. The split is
deliberate: the sampler is hard because its ``ctypes`` struct offsets are
hand-laid and macOS-version-sensitive, and this is hard because of nothing at
all. A reviewer checking offsets and a reviewer checking a kill deadline are
looking for different things, and mixing them hides both.

*The cap is a deadline, not just a comparison.* The fleet spec says a tree that
crosses its limit is killed within :data:`KILL_DEADLINE_SECONDS`. Nothing about
"is this number bigger than that number" enforces that; what enforces it is how
often the number is taken. So :data:`MAX_SAMPLE_INTERVAL_SECONDS` lives here,
beside the deadline it derives from, and
:class:`~nanobot.fleet.supervisor.FleetSupervisor` refuses at construction to be
built with a coarser tick. A supervisor sampling every thirty seconds would pass
every threshold test ever written for it and still miss the deadline by a factor
of six.

*An unreadable tree is not a breach.* :func:`sample` returns ``None`` for a
platform the sampler does not support and for a group walk that failed, and both
cases leave the instance alone. Killing a confined service because the
supervisor could not read a kernel struct would be a far worse failure than
letting one instance run over its cap until the next tick; a tree whose members
have all exited samples as ``0``, which is how "using nothing" stays
distinguishable from "cannot tell".

*The kill is not here.* This module decides; the supervisor acts. Signalling an
instance's tree means addressing its process *group*, which the supervisor
already owns along with the guard that stops a mis-spawned instance's group from
resolving to the supervisor's own. Folding the kill into the policy would drag
that guard along with it, and would race the state write that publishes the
``"memory"`` exit reason. The import direction follows: the supervisor imports
this module and this module never imports the supervisor, which is also what
lets :class:`MemoryCap` be tested with no fleet at all.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from nanobot.fleet.memory import process_group_memory_bytes

#: Bytes in one megabyte, binary. ``memoryLimitMb`` is read the way every other
#: memory limit an operator meets is read — ``ulimit``, container limits, the
#: numbers ``ps`` and Activity Monitor print — so a decimal conversion here would
#: silently make every cap in every fleet file five percent looser than declared.
BYTES_PER_MB = 1_048_576

#: How long the fleet spec allows between a tree crossing its cap and that tree
#: being dead. The whole reason this module has a timing rule at all.
KILL_DEADLINE_SECONDS = 5.0

#: The coarsest interval at which a fleet may be sampled. Derived from the
#: deadline above rather than chosen: the worst case is a tree that crosses its
#: limit one instant after a sample is taken, so it goes unnoticed for a whole
#: interval before the next sample finds it and ``SIGKILL`` (which the kernel
#: does not let a process ignore, delay or handle) ends it. One second leaves
#: four fifths of the budget unspent, which is the margin a loop that also reaps
#: children and writes a state file needs.
MAX_SAMPLE_INTERVAL_SECONDS = 1.0

#: Takes a process group id, returns its tree's resident bytes, or ``None`` when
#: the group cannot be sampled. Injectable so the policy can be driven over
#: known numbers rather than by arranging real memory pressure.
Sampler = Callable[[int], int | None]


@dataclass(frozen=True)
class CapBreach:
    """One instance's tree, caught over its declared limit.

    ``resident_bytes`` is kept beside the limit because the two answer different
    questions: the limit is what the operator declared and the reading is what
    the instance actually did, and an operator raising a cap needs to know by how
    much it was missed.
    """

    name: str
    process_group: int
    limit_mb: int
    resident_bytes: int

    @property
    def limit_bytes(self) -> int:
        """The declared limit in the same units as the reading."""
        return self.limit_mb * BYTES_PER_MB

    @property
    def excess_bytes(self) -> int:
        """How far over the limit the tree was when it was caught."""
        return self.resident_bytes - self.limit_bytes

    def describe(self) -> str:
        """A one-line summary for a log or an error message."""
        return (
            f"instance {self.name}: process tree used "
            f"{self.resident_bytes / BYTES_PER_MB:.1f} MB, over its "
            f"{self.limit_mb} MB limit"
        )


class MemoryCap:
    """Each instance's declared limit, and the sampler that tests it.

    Holds no process handles and sends no signals, so it can be exercised
    standalone: give it a name-to-limit mapping and a function returning numbers,
    and every threshold decision in the fleet is reachable without a fleet.
    """

    def __init__(
        self,
        limits: Mapping[str, int],
        *,
        sample: Sampler = process_group_memory_bytes,
    ) -> None:
        """Bind the declared limits to a way of measuring against them.

        Args:
            limits: instance name to ``memoryLimitMb``.
            sample: how a process group's resident total is read. Defaults to
                :func:`nanobot.fleet.memory.process_group_memory_bytes`.

        Raises:
            ValueError: a limit is not positive. The fleet document already
                requires ``memoryLimitMb > 0``; re-checking here is what keeps
                this class usable — and honest — on its own, since a limit of
                zero would make every instance breach on its first sample and a
                negative one would make the comparison meaningless.
        """
        bad = sorted(name for name, limit in limits.items() if limit <= 0)
        if bad:
            raise ValueError(
                f"memory limits must be positive megabyte counts, but "
                f"{', '.join(bad)} declared one that is not"
            )
        self._limits = dict(limits)
        self._sample = sample

    @property
    def limits(self) -> Mapping[str, int]:
        """Every instance's declared limit in megabytes, read-only."""
        return MappingProxyType(self._limits)

    def limit_bytes(self, name: str) -> int:
        """``name``'s declared limit, in bytes.

        Raises:
            KeyError: no such instance.
        """
        return self._limits[name] * BYTES_PER_MB

    def breach(self, name: str, process_group: int) -> CapBreach | None:
        """Sample ``name``'s tree and report it if it is over its limit.

        Strictly over: a tree sitting exactly on its declared limit has used
        what it was given and no more, and killing it would make every fleet
        file's number mean one byte less than it says.

        Returns:
            The breach, or ``None`` when the tree is within its limit **or**
            could not be sampled. Those two are deliberately not distinguished
            in the return, because the supervisor does the same thing with both:
            an unreadable tree is left running. A sampler that failed once is
            likely to succeed on the next tick, and the cost of waiting one
            interval is bounded, while the cost of killing a healthy confined
            service on the strength of a failed ``ctypes`` call is not.

        Raises:
            KeyError: no such instance, which means the caller and the fleet
                document disagree about who is in this fleet.
        """
        limit_mb = self._limits[name]
        resident = self._sample(process_group)
        if resident is None:
            return None
        if resident <= limit_mb * BYTES_PER_MB:
            return None
        return CapBreach(
            name=name,
            process_group=process_group,
            limit_mb=limit_mb,
            resident_bytes=resident,
        )
