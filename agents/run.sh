#!/usr/bin/env bash
# Task orchestrator — Silpi implements, Viharapala reviews, in sequence.
#
# Run from a plain tmux session OUTSIDE any Claude Code session:
#   tmux new-session -s "Bantu" "bash /path/to/agents/run.sh"
#
# The loop runs until no tasks remain, then sleeps and retries.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
IDLE_INTERVAL=120  # seconds to wait when no tasks are ready

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# ── Guard ─────────────────────────────────────────────────────────────────────
if [ -n "${CLAUDECODE:-}" ]; then
    echo "ERROR: CLAUDECODE is set — this script must not run inside a Claude Code session."
    echo "Open a plain tmux window and run: bash agents/run.sh"
    exit 1
fi

# ── Helpers ───────────────────────────────────────────────────────────────────
slugify() {
    echo "$1" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9]/-/g' \
              | sed 's/--*/-/g' | sed 's/^-//' | sed 's/-$//'
}

review_state() {
    bd state "$1" review 2>/dev/null | tr -d '"' || echo ""
}

run_silpi() {
    local prompt="$1"
    claude \
        --system-prompt "$(cat "$REPO_ROOT/AGENTS.md")" \
        --permission-mode bypassPermissions \
        -p "$prompt"
}

run_viharapala() {
    local prompt="$1"
    claude \
        --system-prompt "$(cat "$SCRIPT_DIR/viharapala/SOUL.md")

---

$(cat "$SCRIPT_DIR/viharapala/AGENTS.md")" \
        --permission-mode bypassPermissions \
        -p "$prompt"
}

# ── Main loop ─────────────────────────────────────────────────────────────────
cd "$REPO_ROOT"
log "Orchestrator started (Silpi + Viharapala)."

while true; do

    # ── 1. Find next ready task ───────────────────────────────────────────────
    TASK_JSON=$(bd ready --json 2>/dev/null \
        | python3 -c "
import sys, json
tasks = json.load(sys.stdin)
print(json.dumps(tasks[0]) if tasks else '')
" 2>/dev/null || echo "")

    if [ -z "$TASK_JSON" ]; then
        log "No tasks ready. Sleeping ${IDLE_INTERVAL}s..."
        sleep "$IDLE_INTERVAL"
        continue
    fi

    TASK_ID=$(echo "$TASK_JSON"    | python3 -c "import sys,json; print(json.load(sys.stdin)['id'])")
    TASK_TITLE=$(echo "$TASK_JSON" | python3 -c "import sys,json; print(json.load(sys.stdin)['title'])")
    BRANCH="feature/$(slugify "$TASK_TITLE")"

    log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    log "Task $TASK_ID: $TASK_TITLE"
    log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # ── 2. Claim ──────────────────────────────────────────────────────────────
    bd update "$TASK_ID" --claim --json
    log "Claimed $TASK_ID."

    # ── 3. Feature branch ─────────────────────────────────────────────────────
    git checkout main
    git pull
    git checkout -b "$BRANCH"
    log "Branch: $BRANCH"

    # ── 4. Silpi / Viharapala loop ────────────────────────────────────────────
    ROUND=1
    while true; do
        TASK_CONTEXT=$(bd show "$TASK_ID" 2>/dev/null || echo "$TASK_JSON")

        # ── Silpi ─────────────────────────────────────────────────────────────
        if [ "$ROUND" -eq 1 ]; then
            log "[Round $ROUND] Silpi implementing $TASK_ID..."
            run_silpi "You are Silpi, working on task $TASK_ID on branch '$BRANCH'.

## Task
$TASK_CONTEXT

## Instructions
1. Implement the task fully.
2. Write unit tests covering the new behaviour.
3. Run \`pytest\` — all tests must pass before committing.
4. Commit all changes using the format: $TASK_ID: <brief description>
5. When done, submit for review:
   bd set-state $TASK_ID review=ready-for-review --reason 'Implementation complete' --json"
        else
            REVIEW_COMMENTS=$(bd comments "$TASK_ID" 2>/dev/null || echo "See bd comments.")
            log "[Round $ROUND] Silpi addressing review comments on $TASK_ID..."
            run_silpi "You are Silpi, addressing review feedback on task $TASK_ID on branch '$BRANCH'.

## Task
$TASK_CONTEXT

## Review comments — fix ALL blocking issues
$REVIEW_COMMENTS

## Instructions
1. Fix every blocking issue in the review comments.
2. Run \`pytest\` — all tests must pass.
3. Commit using the format: $TASK_ID: Address review feedback (round $ROUND)
4. Resubmit for review:
   bd set-state $TASK_ID review=ready-for-review --reason 'Changes addressed' --json"
        fi

        # Guard: ensure review was submitted before handing off
        STATE=$(review_state "$TASK_ID")
        if [ "$STATE" != "ready-for-review" ] && [ "$STATE" != "approved" ] && [ "$STATE" != "changes-required" ]; then
            log "Silpi did not submit for review — submitting now..."
            bd set-state "$TASK_ID" review=ready-for-review --reason "Implementation complete" --json
        fi

        # ── Viharapala ────────────────────────────────────────────────────────
        log "[Round $ROUND] Viharapala reviewing $TASK_ID..."
        run_viharapala "Review task $TASK_ID only. It has been submitted for review on branch '$BRANCH'."

        # Check verdict
        STATE=$(review_state "$TASK_ID")
        case "$STATE" in
            approved)
                log "$TASK_ID approved after $ROUND round(s)."
                break
                ;;
            changes-required)
                log "$TASK_ID needs changes. Handing back to Silpi (round $((ROUND + 1)))..."
                ROUND=$((ROUND + 1))
                ;;
            *)
                log "Viharapala did not set a verdict (state='${STATE:-none}'). Re-running review..."
                ;;
        esac
    done

    # ── 5. Merge to main ──────────────────────────────────────────────────────
    log "Merging $BRANCH to main..."
    git checkout main
    git pull
    git merge --no-ff "$BRANCH" -m "$TASK_ID: Merge $BRANCH"
    git push
    git branch -d "$BRANCH"
    log "Merged and pushed."

    # ── 6. Close ──────────────────────────────────────────────────────────────
    bd close "$TASK_ID" --reason "Approved and merged to main" --json
    log "Task $TASK_ID complete. Moving to next task."

done
