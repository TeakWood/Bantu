# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

**Bantu** (published as `nanobot-ai`) is an ultra-lightweight personal AI agent framework in Python (~4,000 core lines). It supports multiple chat platforms (Telegram, Discord, WhatsApp, Feishu, DingTalk, Slack, Email, QQ, Matrix) and LLM providers (OpenRouter, OpenAI, Anthropic, Qwen, DeepSeek, and more).

## Commands

```bash
# Run tests
pytest
pytest tests/test_specific.py            # single test file
pytest tests/test_specific.py::test_fn   # single test function

# Lint / format
ruff check nanobot/
ruff format nanobot/

# Run the agent (development)
pip install -e .
nanobot onboard    # first-time setup
nanobot agent      # interactive chat
nanobot gateway    # run as gateway server

# Count core agent lines
bash core_agent_lines.sh
```

## Architecture

The project follows a clean adapter pattern with three main layers:

**Core engine** (`nanobot/agent/`):
- `loop.py` — main `AgentLoop`: receives messages from the bus, builds context, calls the LLM, executes tool calls, sends responses back
- `context.py` — builds prompt context from history, memory, and skills
- `memory.py` — memory store with consolidation and offset tracking
- `skills.py` — loads skill definitions as template text
- `subagent.py` — manages spawned subagents
- `tools/` — tool implementations (filesystem, shell, web, cron, MCP, message) plus registry

**Adapters**:
- `nanobot/channels/` — one file per platform; `base.py` defines the interface, `manager.py` routes messages
- `nanobot/providers/` — LLM providers (litellm, openai-codex, custom); `registry.py` selects at runtime

**Infrastructure**:
- `nanobot/bus/` — async `MessageBus` with `InboundMessage` / `OutboundMessage` events
- `nanobot/session/` — session persistence and conversation history
- `nanobot/config/schema.py` — Pydantic v2 models for all config (supports snake_case and camelCase)
- `nanobot/cli/commands.py` — all CLI commands via Typer
- `nanobot/cron/` — scheduled task service
- `nanobot/skills/` — built-in skills (clawhub, cron, github, memory, summarize, weather, tmux, skill-creator)
- `nanobot/templates/` — agent prompt templates (AGENTS.md, SOUL.md, TOOLS.md, etc.)

**Bridge**: `bridge/` is a separate Node.js/TypeScript service for WhatsApp.

## Issue Tracking (beads / bd)

This repo uses **bd** (beads) for all issue tracking — not markdown TODOs.

```bash
bd ready --json                  # find available work
bd show <id>                     # view issue details
bd update <id> --claim --json    # claim work
bd create "Title" --description="..." -t bug|feature|task -p 0-4 --json
bd close <id> --reason "Done"    # complete work
bd sync                          # sync with git
```

Issue types: `bug`, `feature`, `task`, `epic`, `chore`. Priorities: 0=Critical … 4=Backlog.
Link discovered work: `--deps discovered-from:<parent-id>`.

### Code Review States

Review state is tracked on the `review` dimension via `bd set-state`:

```bash
# Submit a task for review (Silpi)
bd set-state <id> review=ready-for-review --reason "Ready for review"

# Query tasks awaiting review
bd query "label=review:ready-for-review" --json

# Reviewer verdicts
bd set-state <id> review=changes-required --reason "See review comment"
bd set-state <id> review=approved --reason "LGTM"

# Add a review comment
bd comments add <id> "<review text>"
```

### Viharapala — Automated Code Reviewer

`agents/viharapala/` contains the workspace for **Viharapala**, an autonomous
code reviewer agent. It picks up every `review:ready-for-review` task, runs
quality gates, and posts a structured review comment with a verdict.

Viharapala is invoked by `agents/run.sh` after each Silpi implementation round. It never modifies source code — it only reads, comments, and sets review states.

**Workspace layout:**
```
agents/viharapala/
├── SOUL.md          # Identity and personality
├── AGENTS.md        # Full review loop instructions
└── skills/
    └── bd-review/
        └── SKILL.md # bd review state commands (always loaded)
```

## Epic Workflow

Epics are **design tasks first**. Never write implementation code until the design is validated by the author.

### Phase 1 — Design (do not skip)

1. **Gather requirements**: Read the epic description thoroughly. Ask clarifying questions until the scope is unambiguous.
2. **Propose architecture**: Write up the design — data flow, affected modules, API changes, config changes, edge cases. Be specific enough that reviewers can spot gaps. Post it as a bd comment and submit for review:
   ```bash
   bd comments add <epic-id> "<design proposal>"
   bd set-state <epic-id> review=ready-for-review --reason "Design ready for review" --json
   ```
3. **Viharapala reviews first**: Viharapala evaluates the design (not code — no quality gates) and sets either `review=viharapala-approved` or `review=changes-required`. If changes are required, revise the design comment and resubmit.
4. **Author review**: Once Viharapala sets `review=viharapala-approved`, **Navakanth Gandavarapu** manually reviews the design and sets the final approval:
   ```bash
   bd set-state <epic-id> review=approved --reason "Design approved" --json
   ```
5. **Do not proceed** to Phase 2 until `review=approved` is set by the author. Silpi must poll or wait — never skip this gate.

### Phase 2 — Break down into feature tasks

Only after the author approves the design:

1. **Decompose** the epic into `feature` issues in bd. Each feature task must:
   - Be small and logically self-contained
   - Leave Bantu in a working, usable state when complete (no half-broken intermediary states)
   - Include unit tests as part of its definition of done
2. **Link dependencies**: if task B requires task A to be complete, set `--deps <A-id>` on B when creating it.
3. **Create the issues**:
   ```bash
   bd create "Feature: <name>" --description="..." -t feature -p <priority> --deps discovered-from:<epic-id> --json
   # For dependent tasks:
   bd create "Feature: <name>" --description="..." -t feature -p <priority> --deps discovered-from:<epic-id> <blocking-id> --json
   ```

### Phase 3 — Implement each feature task

Run `agents/run.sh` from a **plain tmux session outside any Claude Code session**. It orchestrates Silpi and Viharapala in sequence until all tasks are done:

```bash
# Start a plain tmux session and run the orchestrator
tmux new-session -s "Bantu" "bash $(pwd)/agents/run.sh"
```

The script will exit with an error if `CLAUDECODE` is set (i.e. if accidentally run from inside a Claude Code session).

**Per-task flow:**
1. Claim task → create `feature/<title>` branch
2. **Silpi** (fresh `claude -p`): implement, test, commit, submit for review
3. **Viharapala** (fresh `claude -p`): review, comment, set verdict
4. If `changes-required` → back to Silpi with review comments (new session)
5. If `approved` (by both Viharapala + author for epics; by Viharapala for feature tasks) → squash merge to main, push, close task
6. Repeat for next task

### Rules

- **Never start coding an epic without author sign-off on the design.**
- Feature tasks must keep Bantu functional — no commits that break the agent loop, CLI, or an active channel.
- Tests are not optional; every feature task ships with tests.

## Shell Safety

Many systems alias `cp`, `mv`, `rm` with `-i` (interactive), which causes hangs. Always use:

```bash
cp -f   mv -f   rm -f   rm -rf   cp -rf
```

## Session Completion

Before ending a session: close/update issues, run quality gates if code changed, then:

```bash
git pull --rebase && bd sync && git push
```

Work is not complete until `git push` succeeds.
