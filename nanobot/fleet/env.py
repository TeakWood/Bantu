"""Build the environment one fleet instance is allowed to see.

The supervisor's own environment is the union of every credential on the
machine: provider keys, web-search keys, whatever the operator exported before
typing ``nanobot fleet start``. A child that inherits it inherits all of them,
which is exactly the convention-only separation the fleet exists to replace —
``docs/multiple-instances.md`` instances share their whole environment today.

So an instance is launched under ``/usr/bin/env -i`` with nothing but what this
module returns: four variables a process needs in order to run at all, plus the
variables that instance's own fleet entry names. Nothing else crosses, and in
particular nothing another instance's entry names.

Two details make this load-bearing rather than a dict comprehension.

*A missing variable must stay missing.* nanobot's config loader resolves
``${VAR}`` references when it reads a config file and raises
``ConfigLoadError(kind="missing_env")`` when the referenced variable is unset
(``config/loader.py``). That is the behaviour the fleet wants: an instance whose
config reaches for a key its entry never listed fails to start, loudly, at
launch. Forwarding an unset name as ``""`` would defeat it — the reference would
resolve to empty and the instance would come up and then talk to a provider with
a blank API key, failing later and further away. The same applies to keys read
straight from ``os.environ``, bypassing config entirely. Absent is a meaningful
state, so absent is preserved; that rule covers the base variables too, since
inventing a value the supervisor does not have is the same mistake.

*Names are re-checked here.* ``nanobot.fleet.config`` already constrains ``env``
entries to :data:`~nanobot.fleet.config.ENV_NAME_PATTERN`, which is what stops an
operator writing ``"OPENAI_API_KEY=sk-…"`` into a file meant to hold no secrets.
The check is repeated on this side because the result is rendered into ``env -i``
argv words of the form ``NAME=value``: a name containing ``=`` would smuggle in a
second variable the fleet document never declared. The invariant belongs where it
is relied upon, not only where the document happens to be parsed.

This module reads ``os.environ`` and nothing else: no filesystem, no config
loading, no global state.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable, Mapping

from nanobot.fleet.config import ENV_NAME_PATTERN

#: The whole base environment, in the order it is emitted.
#:
#: ``PATH`` so the instance can find the executables its shell tool runs, ``HOME``
#: because Python and most libraries derive per-user paths from it, ``LANG`` so
#: text decoding does not depend on the launcher's locale, and ``TMPDIR`` because
#: on macOS it names a per-user directory that the instance's Seatbelt profile
#: allows — falling back to a shared ``/tmp`` would hand every instance a
#: directory its peers can also write.
#:
#: Deliberately absent: ``TERM`` (an instance is not attached to a terminal) and
#: every provider or search credential (those are per-instance by declaration).
MINIMAL_ENV_NAMES: tuple[str, ...] = ("PATH", "HOME", "LANG", "TMPDIR")

_ENV_NAME_RE = re.compile(ENV_NAME_PATTERN)


class InstanceEnvError(ValueError):
    """An instance's declared ``env`` list cannot be turned into an environment.

    Raised only for a name that is not shaped like an environment variable name.
    The rejected name is carried unaltered because it came from the fleet
    document, where values are never meant to appear; a caller that renders it is
    echoing a declaration, not a credential.
    """

    def __init__(self, name: str) -> None:
        super().__init__(
            f"{name!r} is not a valid environment variable name "
            f"(must match {ENV_NAME_PATTERN})"
        )
        self.name = name


def instance_environment(
    env_names: Iterable[str] = (),
    *,
    source: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return the complete environment for one instance.

    Args:
        env_names: the variable *names* from this instance's fleet entry. Only
            this instance's list may be passed; the isolation the fleet promises
            is precisely that a peer's list has no effect here.
        source: the supervisor environment to draw values from. Defaults to
            ``os.environ``; an explicit mapping exists so a caller can launch
            from a snapshot rather than from whatever the process environment has
            become by then.

    Returns:
        A fresh dict holding the base variables present in *source*, in
        :data:`MINIMAL_ENV_NAMES` order, followed by the requested names in
        declaration order. A name absent from *source* is omitted; the result
        never contains a variable that was not set, and never an empty value
        standing in for one.

    Raises:
        InstanceEnvError: a requested name is not a valid variable name. Every
            name is checked before any value is read, so a refusal happens
            without touching the supervisor's credentials at all.
    """
    requested = tuple(env_names)
    for name in requested:
        if not _ENV_NAME_RE.fullmatch(name):
            raise InstanceEnvError(name)

    environ = os.environ if source is None else source
    env: dict[str, str] = {}
    for name in (*MINIMAL_ENV_NAMES, *requested):
        _put(env, environ, name)
    return env


def _put(env: dict[str, str], source: Mapping[str, str], name: str) -> None:
    """Copy ``name`` from *source* if it is set there, otherwise leave it unset."""
    value = source.get(name)
    if value is not None:
        env[name] = value
