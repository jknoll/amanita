# Handoff Skill

Prepares the current session state for pickup on another host running Claude Code.

## When to Use

Use `/handoff` when you want to:
- End a Claude Code session on the current host
- Continue work on a different machine
- Document the current state for future reference

## Instructions

When this skill is invoked:

1. **Read Project Context**
   - Read `CLAUDE.md` for project guidelines
   - Check `git status` and `git log` for current state

2. **Summarize Session Work**
   - What was accomplished in this session
   - Key findings or discoveries
   - Any issues encountered and how they were resolved

3. **Write HANDOFF.md**
   Create or update `HANDOFF.md` in the project root with:
   - **Date**: Current date
   - **Last Commit**: Most recent commit hash and message
   - **Session Summary**: Brief overview of what was done
   - **What Was Done**: Detailed list of completed work
   - **Key Findings**: Important discoveries or results
   - **Files Created/Modified**: List of changed files
   - **Environment Notes**: Any setup or dependency information
   - **Next Steps**: Suggested follow-up work
   - **Git Status**: Current branch and uncommitted changes

4. **Commit and Push**
   - Stage `HANDOFF.md`
   - Commit with message: "Add session handoff: [brief description]"
   - Push to current branch

5. **Confirm Handoff**
   - Display the commit hash
   - Confirm push was successful
   - Remind user to run `/pickup` on the new host

## Example Output

```
Handoff complete!

Committed: abc1234 - Add session handoff: validation notebook implementation
Pushed to: main

To continue on another host:
1. Pull the latest changes
2. Run /pickup to restore context
```
