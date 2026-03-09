#!/usr/bin/env python3
"""Task orchestrator — Silpi implements, Viharapala reviews, in sequence.

Run from a plain terminal OUTSIDE any Claude Code session:
    python agents/run.py

The loop runs until no tasks remain, then sleeps and retries.
"""

import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import anyio
from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    TextBlock,
    query,
)

SCRIPT_DIR = Path(__file__).parent.resolve()
REPO_ROOT = SCRIPT_DIR.parent.resolve()
IDLE_INTERVAL = 120  # seconds to wait when no tasks are ready
TASK_STATE_FILE = REPO_ROOT / ".claude" / "current-task.json"
EPIC_BREAKDOWN_FILE = REPO_ROOT / ".claude" / "epic-breakdown.json"


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def slugify(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]", "-", text)
    text = re.sub(r"-+", "-", text)
    return text.strip("-")


def run_cmd(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def run_cmd_output(cmd: list[str]) -> str:
    result = subprocess.run(cmd, check=False, text=True, capture_output=True, cwd=REPO_ROOT)
    return result.stdout.strip()


def review_state(task_id: str) -> str:
    out = run_cmd_output(["bd", "state", task_id, "review"])
    return out.strip('"')


def breakdown_state(epic_id: str) -> str:
    out = run_cmd_output(["bd", "state", epic_id, "breakdown"])
    return out.strip('"')


# ── Task state file ────────────────────────────────────────────────────────────

def save_task_state(task: dict, branch: str) -> None:
    TASK_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    TASK_STATE_FILE.write_text(json.dumps({"task": task, "branch": branch}))


def clear_task_state() -> None:
    if TASK_STATE_FILE.exists():
        TASK_STATE_FILE.unlink()


def load_task_state() -> tuple[dict | None, str | None]:
    if not TASK_STATE_FILE.exists():
        return None, None
    try:
        data = json.loads(TASK_STATE_FILE.read_text())
        return data.get("task"), data.get("branch")
    except (json.JSONDecodeError, KeyError):
        return None, None


# ── Epic breakdown state file ──────────────────────────────────────────────────

def save_epic_breakdown(epic: dict) -> None:
    EPIC_BREAKDOWN_FILE.parent.mkdir(parents=True, exist_ok=True)
    EPIC_BREAKDOWN_FILE.write_text(json.dumps({"epic": epic}))


def clear_epic_breakdown() -> None:
    if EPIC_BREAKDOWN_FILE.exists():
        EPIC_BREAKDOWN_FILE.unlink()


def load_epic_breakdown() -> dict | None:
    if not EPIC_BREAKDOWN_FILE.exists():
        return None
    try:
        data = json.loads(EPIC_BREAKDOWN_FILE.read_text())
        return data.get("epic")
    except (json.JSONDecodeError, KeyError):
        return None


# ── Resume helpers ─────────────────────────────────────────────────────────────

def classify_resume_state(rev_state: str) -> str:
    """Map bd review dimension value → orchestrator action."""
    if not rev_state:
        return "silpi_implement"
    if rev_state == "ready-for-review":
        return "viharapala_review"
    if rev_state == "changes-required":
        return "silpi_address"
    if rev_state == "approved":
        return "merge"
    return "silpi_implement"


def detect_next_round(branch: str) -> int:
    """Count 'Address review feedback' commits in branch to find next round number."""
    out = run_cmd_output(["git", "log", f"main..{branch}", "--oneline"])
    if not out:
        return 1
    count = sum(1 for line in out.splitlines() if "Address review feedback" in line)
    return count + 1


def find_resumable_task() -> tuple[dict | None, str | None, str, int]:
    """Check for a crash-interrupted task in priority order.

    Returns (task, branch, state, round_num) or (None, None, "none", 1).
    """
    # Priority 1 — state file
    task, branch = load_task_state()
    if task and branch:
        task_id = task["id"]
        rev = review_state(task_id)
        state = classify_resume_state(rev)
        round_num = detect_next_round(branch) if state != "silpi_implement" else 1
        log(f"Resume: state file → task {task_id}, branch={branch}, state={state}, round={round_num}")
        return task, branch, state, round_num

    # Priority 2 — current feature branch
    current_branch = run_cmd_output(["git", "branch", "--show-current"])
    if current_branch.startswith("feature/"):
        try:
            result = subprocess.run(
                ["bd", "query", "status=in_progress", "--json"],
                check=False, text=True, capture_output=True, cwd=REPO_ROOT,
            )
            in_progress = json.loads(result.stdout or "[]")
        except (json.JSONDecodeError, FileNotFoundError):
            in_progress = []
        for t in in_progress:
            if f"feature/{slugify(t['title'])}" == current_branch:
                rev = review_state(t["id"])
                state = classify_resume_state(rev)
                round_num = detect_next_round(current_branch)
                log(f"Resume: current branch → task {t['id']}, state={state}, round={round_num}")
                return t, current_branch, state, round_num

    # Priority 3 — any ready-for-review task (Viharapala crashed)
    try:
        result = subprocess.run(
            ["bd", "query", "label=review:ready-for-review", "--json"],
            check=False, text=True, capture_output=True, cwd=REPO_ROOT,
        )
        rfr_tasks = json.loads(result.stdout or "[]")
    except (json.JSONDecodeError, FileNotFoundError):
        rfr_tasks = []
    if rfr_tasks:
        t = rfr_tasks[0]
        branch = f"feature/{slugify(t['title'])}"
        round_num = detect_next_round(branch)
        log(f"Resume: ready-for-review → task {t['id']}, branch={branch}, round={round_num}")
        return t, branch, "viharapala_review", round_num

    return None, None, "none", 1


# ── Agent runners ──────────────────────────────────────────────────────────────

async def run_agent(system_prompt: str, prompt: str) -> None:
    """Run a Claude agent session and stream output to stdout."""
    async for message in query(
        prompt=prompt,
        options=ClaudeAgentOptions(
            system_prompt=system_prompt,
            permission_mode="bypassPermissions",
            cwd=str(REPO_ROOT),
        ),
    ):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    print(block.text, end="", flush=True)
        elif isinstance(message, ResultMessage):
            if message.result:
                print(message.result, flush=True)


async def run_silpi(prompt: str) -> None:
    soul = (SCRIPT_DIR / "silpi" / "SOUL.md").read_text()
    agents_md = (SCRIPT_DIR / "silpi" / "AGENTS.md").read_text()
    system_prompt = f"{soul}\n\n---\n\n{agents_md}"
    await run_agent(system_prompt, prompt)


async def run_viharapala(prompt: str) -> None:
    soul = (SCRIPT_DIR / "viharapala" / "SOUL.md").read_text()
    agents_md = (SCRIPT_DIR / "viharapala" / "AGENTS.md").read_text()
    system_prompt = f"{soul}\n\n---\n\n{agents_md}"
    await run_agent(system_prompt, prompt)


# ── Epic breakdown ─────────────────────────────────────────────────────────────

async def run_epic_breakdown(epic: dict) -> None:
    """Invoke Silpi to decompose an approved epic into feature tasks."""
    epic_id = epic["id"]
    log(f"Breaking down epic {epic_id}: {epic['title']}")
    save_epic_breakdown(epic)

    epic_context = run_cmd_output(["bd", "show", epic_id]) or json.dumps(epic)
    epic_comments = run_cmd_output(["bd", "comments", epic_id]) or "No comments yet."

    await run_silpi(
        f"You are Silpi, breaking down approved epic {epic_id} into feature tasks.\n\n"
        f"## Epic\n{epic_context}\n\n"
        f"## Design proposal (from comments)\n{epic_comments}\n\n"
        f"## Instructions\n"
        f"Follow the 'Epic Breakdown' section in your AGENTS.md exactly.\n"
        f"The epic ID is {epic_id}.\n"
        f"When all tasks are created, signal completion:\n"
        f"  bd set-state {epic_id} breakdown=complete --reason 'Feature tasks created' --json"
    )

    if breakdown_state(epic_id) == "complete":
        clear_epic_breakdown()
        log(f"Epic {epic_id} breakdown complete.")
    else:
        log(f"Warning: epic {epic_id} breakdown incomplete — will retry on next run.")


# ── Core task loop ─────────────────────────────────────────────────────────────

async def run_task_loop(task: dict, branch: str, start_round: int, start_state: str) -> None:
    """Run the Silpi/Viharapala/merge loop for a single task.

    start_state: "silpi_implement" | "silpi_address" | "viharapala_review" | "merge"
    """
    task_id = task["id"]
    round_num = start_round
    state = start_state

    while True:
        task_context = run_cmd_output(["bd", "show", task_id]) or json.dumps(task)

        # ── Silpi ─────────────────────────────────────────────────────────────
        if state == "silpi_implement":
            log(f"[Round {round_num}] Silpi implementing {task_id}...")
            await run_silpi(
                f"You are Silpi, working on task {task_id} on branch '{branch}'.\n\n"
                f"## Task\n{task_context}\n\n"
                f"## Instructions\n"
                f"1. Implement the task fully.\n"
                f"2. Write unit tests covering the new behaviour.\n"
                f"3. Run `pytest` — all tests must pass before committing.\n"
                f"4. Commit all changes using the format: {task_id}: <brief description>\n"
                f"5. When done, submit for review:\n"
                f"   bd set-state {task_id} review=ready-for-review --reason 'Implementation complete' --json"
            )
            # Guard: ensure review was submitted before handing off
            state = review_state(task_id)
            if state not in ("ready-for-review", "approved", "changes-required"):
                log("Silpi did not submit for review — submitting now...")
                run_cmd([
                    "bd", "set-state", task_id, "review=ready-for-review",
                    "--reason", "Implementation complete", "--json",
                ])
            state = "viharapala_review"

        elif state == "silpi_address":
            review_comments = run_cmd_output(["bd", "comments", task_id]) or "See bd comments."
            log(f"[Round {round_num}] Silpi addressing review comments on {task_id}...")
            await run_silpi(
                f"You are Silpi, addressing review feedback on task {task_id} on branch '{branch}'.\n\n"
                f"## Task\n{task_context}\n\n"
                f"## Review comments — fix ALL blocking issues\n{review_comments}\n\n"
                f"## Instructions\n"
                f"1. Fix every blocking issue in the review comments.\n"
                f"2. Run `pytest` — all tests must pass.\n"
                f"3. Commit using the format: {task_id}: Address review feedback (round {round_num})\n"
                f"4. Resubmit for review:\n"
                f"   bd set-state {task_id} review=ready-for-review --reason 'Changes addressed' --json"
            )
            # Guard: ensure review was submitted before handing off
            state = review_state(task_id)
            if state not in ("ready-for-review", "approved", "changes-required"):
                log("Silpi did not submit for review — submitting now...")
                run_cmd([
                    "bd", "set-state", task_id, "review=ready-for-review",
                    "--reason", "Implementation complete", "--json",
                ])
            state = "viharapala_review"

        # ── Viharapala ────────────────────────────────────────────────────────
        if state == "viharapala_review":
            log(f"[Round {round_num}] Viharapala reviewing {task_id}...")
            await run_viharapala(
                f"Review task {task_id} only. It has been submitted for review on branch '{branch}'."
            )

            verdict = review_state(task_id)
            if verdict == "approved":
                log(f"{task_id} approved after {round_num} round(s).")
                state = "merge"
            elif verdict == "changes-required":
                log(f"{task_id} needs changes. Handing back to Silpi (round {round_num + 1})...")
                round_num += 1
                state = "silpi_address"
                continue
            else:
                log(f"Viharapala did not set a verdict (state='{verdict or 'none'}'). Re-running review...")
                continue

        # ── Merge ─────────────────────────────────────────────────────────────
        if state == "merge":
            branch_exists = bool(run_cmd_output(["git", "branch", "--list", branch]))
            if branch_exists:
                log(f"Merging {branch} to main...")
                run_cmd(["git", "checkout", "main"])
                run_cmd(["git", "pull"])
                run_cmd(["git", "merge", "--squash", branch])
                run_cmd(["git", "commit", "-m", f"{task_id}: Merge {branch}"])
                run_cmd(["git", "push"])
                run_cmd(["git", "branch", "-D", branch])
                log("Merged and pushed.")
            else:
                log(f"Branch {branch} not found locally — skipping merge (already merged).")

            try:
                run_cmd(["bd", "close", task_id, "--reason", "Approved and merged to main", "--json"])
            except subprocess.CalledProcessError:
                log(f"Warning: could not close task {task_id} (may already be closed).")
            clear_task_state()
            log(f"Task {task_id} complete. Moving to next task.")
            return


# ── Main ───────────────────────────────────────────────────────────────────────

async def main() -> None:
    # ── Guard ─────────────────────────────────────────────────────────────────
    if os.environ.get("CLAUDECODE"):
        print(
            "ERROR: CLAUDECODE is set — this script must not run inside a Claude Code session.\n"
            "Open a plain tmux window and run: python agents/run.py",
            file=sys.stderr,
        )
        sys.exit(1)

    log("Orchestrator started (Silpi + Viharapala).")

    # ── Crash recovery: epic breakdown ───────────────────────────────────────
    resume_epic = load_epic_breakdown()
    if resume_epic:
        if breakdown_state(resume_epic["id"]) != "complete":
            log(f"Resuming epic breakdown for {resume_epic['id']}...")
            await run_epic_breakdown(resume_epic)
        else:
            clear_epic_breakdown()

    # ── Crash recovery: task implementation ──────────────────────────────────
    resume_task, resume_branch, resume_state, resume_round = find_resumable_task()
    if resume_task:
        log(f"Resuming task {resume_task['id']} (state={resume_state}, round={resume_round}).")
        current_branch = run_cmd_output(["git", "branch", "--show-current"])
        if current_branch != resume_branch:
            if bool(run_cmd_output(["git", "branch", "--list", resume_branch])):
                run_cmd(["git", "checkout", resume_branch])
            else:
                run_cmd(["git", "checkout", "main"])
                run_cmd(["git", "pull"])
                run_cmd(["git", "checkout", "-b", resume_branch])
        await run_task_loop(resume_task, resume_branch, resume_round, resume_state)

    while True:

        # ── 0. Check for epics awaiting author review ─────────────────────────
        try:
            result = subprocess.run(
                ["bd", "query", "label=review:viharapala-approved", "--json"],
                check=False, text=True, capture_output=True, cwd=REPO_ROOT,
            )
            pending_epics = json.loads(result.stdout or "[]")
        except (json.JSONDecodeError, FileNotFoundError):
            pending_epics = []

        epic_count = sum(1 for t in pending_epics if t.get("type") == "epic")
        if epic_count > 0:
            log("━" * 40)
            log(f"AUTHOR REVIEW REQUIRED — {epic_count} epic(s) approved by Viharapala, awaiting your sign-off.")
            log("━" * 40)
            for t in pending_epics:
                if t.get("type") == "epic":
                    print(f"  Epic {t['id']}: {t['title']}")
            log("")
            log("For each epic above:")
            log("  1. Review the design proposal:")
            log("       bd show <id>")
            log("       bd comments <id>")
            log("")
            log("  2a. Approve the design:")
            log("       bd set-state <id> review=approved --reason 'Design approved' --json")
            log("")
            log("  2b. Request changes:")
            log("       bd set-state <id> review=changes-required --reason '<what to fix>' --json")
            log("")
            log("Re-run this script once you have reviewed all pending epics.")
            sys.exit(0)

        # ── 0b. Break down author-approved epics ──────────────────────────────
        try:
            result = subprocess.run(
                ["bd", "query", "label=review:approved", "--json"],
                check=False, text=True, capture_output=True, cwd=REPO_ROOT,
            )
            approved_tasks = json.loads(result.stdout or "[]")
            approved_epics = [
                t for t in approved_tasks
                if t.get("type") == "epic" and breakdown_state(t["id"]) != "complete"
            ]
        except (json.JSONDecodeError, FileNotFoundError):
            approved_epics = []

        if approved_epics:
            for epic in approved_epics:
                await run_epic_breakdown(epic)
            continue  # restart loop so newly created tasks are picked up

        # ── 1. Find next ready task ────────────────────────────────────────────
        try:
            result = subprocess.run(
                ["bd", "ready", "--json"],
                check=False, text=True, capture_output=True, cwd=REPO_ROOT,
            )
            tasks = json.loads(result.stdout or "[]")
            task = tasks[0] if tasks else None
        except (json.JSONDecodeError, FileNotFoundError):
            task = None

        if not task:
            log(f"No tasks ready. Sleeping {IDLE_INTERVAL}s...")
            await anyio.sleep(IDLE_INTERVAL)
            continue

        task_id: str = task["id"]
        task_title: str = task["title"]
        branch = f"feature/{slugify(task_title)}"

        log("━" * 40)
        log(f"Task {task_id}: {task_title}")
        log("━" * 40)

        # ── 2. Claim ──────────────────────────────────────────────────────────
        run_cmd(["bd", "update", task_id, "--claim", "--json"])
        log(f"Claimed {task_id}.")
        save_task_state(task, branch)

        # ── 3. Feature branch ─────────────────────────────────────────────────
        run_cmd(["git", "checkout", "main"])
        run_cmd(["git", "pull"])
        run_cmd(["git", "checkout", "-b", branch])
        log(f"Branch: {branch}")

        # ── 4. Silpi / Viharapala / merge loop ────────────────────────────────
        await run_task_loop(task, branch, 1, "silpi_implement")


if __name__ == "__main__":
    anyio.run(main)
