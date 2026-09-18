@AGENTS.md


<!-- BEGIN BEADS INTEGRATION v:1 profile:minimal hash:ca08a54f -->
## Beads Issue Tracker

This project uses **bd (beads)** for issue tracking. Run `bd prime` to see full workflow context and commands.

### Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --claim  # Claim work
bd close <id>         # Complete work
```

### Rules

- Use `bd` for ALL task tracking — do NOT use TodoWrite, TaskCreate, or markdown TODO lists
- Run `bd prime` for detailed command reference and session close protocol
- Use `bd remember` for persistent knowledge — do NOT use MEMORY.md files

## Session Completion

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   bd dolt push
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds
<!-- END BEADS INTEGRATION -->

## SHRENI INTEGRATION

This project is managed by Shreni. The Sthapathi daemon picks up beads issues and
implements them via autonomous agents (Silpi, Viharapala, Parikshaka).

**If your system prompt assigns you a Silpi/Viharapala/Parikshaka role for a
specific bead, this section does NOT apply to you** — do your assigned job
(implement / review / analyze) with your tools. The rules below govern
interactive human sessions only.

**Interactive sessions: task producer only.**
Create beads issues for the daemon to implement — do NOT implement tasks yourself.

Prohibited in interactive sessions:
  bd update --claim            Sthapathi claims tasks, not interactive agents
  bd close                     Sthapathi closes tasks on completion
  git checkout -b / git branch Sthapathi owns all bead-* branches

Useful commands:
  shreni status --all          Show all kshetra states
  shreni agents                Show live agent activity
  shreni logs --kshetra <id>   Round-by-round agent logs
  shreni pause --kshetra <id>  Pause task pickup
  shreni resume --kshetra <id> Resume task pickup

### Toolchain config sync

Shreni runs build/test/lint from the pointers in `.shreni/kshetra.yaml` (stack.*),
not by re-discovering your toolchain. Whenever you add or change a toolchain
config file — a new test runner (vitest/jest/pytest), linter (eslint), tsconfig,
a new package.json/Makefile script, or you switch package managers — update the
matching pointer in `.shreni/kshetra.yaml` in the same change:

  stack.buildCommand   the build/compile gate (e.g. `pnpm build`)
  stack.testRunner     the test command (e.g. `pnpm test`)
  stack.lintCommand    the lint gate (e.g. `pnpm lint`); omit to skip lint

Prefer pointing at a project script (`pnpm test`) over duplicating globs. The
escape hatches stack.testFileGlobs / stack.failCountPattern are for non-standard
setups only — set them only when the harness must find tests WITHOUT running the
runner. A stale pointer means Shreni runs the wrong gate.
