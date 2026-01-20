# Pickup Skill

Restores session context from a previous handoff on another host.

## When to Use

Use `/pickup` when you want to:
- Continue work that was handed off from another machine
- Restore context from a previous Claude Code session
- Understand the current state of the project after a break

## Instructions

When this skill is invoked:

1. **Sync Repository**
   - Run `git pull` to get latest changes
   - Check for any merge conflicts

2. **Read Handoff Context**
   - Read `HANDOFF.md` for previous session state
   - Note the date and last commit of the handoff

3. **Read Project Guidelines**
   - Read `CLAUDE.md` for project-specific instructions
   - Note any environment setup requirements

4. **Summarize State to User**
   Present a clear summary:
   - When the handoff occurred
   - What was accomplished in the previous session
   - Key findings or important context
   - Suggested next steps
   - Current git status

5. **Environment Check**
   - Check if required dependencies are available
   - Note any environment differences from the handoff
   - Suggest setup steps if needed

6. **Ask for Direction**
   - Present the suggested next steps from HANDOFF.md
   - Ask the user what they'd like to work on

## Example Output

```
Pickup from handoff on 2026-01-19

Last session accomplished:
- Created validation notebook for multi-task BEiT model
- Discovered 25% accuracy drop vs stock baseline at species level
- Generated HTML report and visualizations

Key findings:
- Species accuracy: 45.43% (vs 70.2% baseline)
- Amanita phalloides detection needs improvement

Suggested next steps:
1. Investigate training logs
2. Try different loss weighting
3. Evaluate on test set

Current status: On branch main, up to date with origin

What would you like to work on?
```

## Notes

- If `HANDOFF.md` doesn't exist, inform the user and offer to read `CLAUDE.md` instead
- If there are uncommitted changes, warn the user before pulling
- The pickup doesn't need to be committed - it's a read-only operation
