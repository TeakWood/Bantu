---
name: bd-review
description: "Manage code review states on bd issues using the review dimension."
metadata: {"nanobot": {"always": true, "requires": {"bins": ["bd"]}}}
---

# bd Review State Skill

Use the `review` dimension on bd issues to track code review status.

## States

| State | Meaning |
|---|---|
| `review:ready-for-review` | Author has submitted the task for review |
| `review:changes-required` | Reviewer found blocking issues |
| `review:viharapala-approved` | Viharapala approved (epics only) — awaiting author's final review |
| `review:approved` | Final approval granted — safe to merge/close |

> For **feature tasks**: Viharapala sets `approved` directly.
> For **epics**: Viharapala sets `viharapala-approved`; Navakanth Gandavarapu sets the final `approved`.

## Commands

**Find all tasks awaiting review:**
```bash
bd query "label=review:ready-for-review" --json
```

**Mark as needing changes:**
```bash
bd set-state <id> review=changes-required --reason "<brief reason>"
```

**Approve (feature tasks — Viharapala final approval):**
```bash
bd set-state <id> review=approved --reason "LGTM"
```

**Approve (epics — Viharapala first-pass only, awaits author):**
```bash
bd set-state <id> review=viharapala-approved --reason "Design LGTM — ready for author review"
```

**Submit a task for review (used by implementers, not reviewers):**
```bash
bd set-state <id> review=ready-for-review --reason "Ready for review"
```

**Add a review comment:**
```bash
bd comments add <id> "<review text>"
```

**Read existing review comments:**
```bash
bd comments <id> --json
```
