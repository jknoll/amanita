# Session Handoff

**Date**: 2026-01-22
**Last Commit**: `14cde7e Merge pull request #2 from jknoll/fix/readme-formatting`

## Session Summary

This session focused on syncing experiment artifacts from the Strong Compute ISC cluster to local storage for analysis. Created documentation for the artifact sync process and successfully transferred all missing checkpoints.

## What Was Done

### 1. Created Artifact Sync Documentation (`ARTIFACT-SYNC.md`)
- Documented the full process for syncing artifacts from ISC to local machine
- Covers prerequisites (WireGuard VPN, SSH keys, running container)
- Step-by-step instructions for triggering sync and SCP commands
- Troubleshooting section for common issues

### 2. Synced Missing Checkpoints
Transferred checkpoints from remote ISC workstation (`192.168.127.71:52134`) to local storage (`/Volumes/Extra FAT/Amanita-Validation/`):

**exp-organized-valley-fig-260101**: 27 new checkpoint directories added (28 total)
- Checkpoints 2, 4, 6, 8, 12, 15, 17, 19, 21, 23, 25, 26, 29, 31, 33, 35, 37, 38, 42, 44, 46, 48, 50, 53, 56, 58, 60
- Note: Some directories (8, 21, 23, 37, 46, 48, 50) are empty on remote - incomplete training runs

**exp-safe-tabby-dirigible-251229**: 1 checkpoint added
- AtomicDirectory_checkpoint_2/test_checkpoint.pt

### 3. Merged PRs
- PR #2 (fix/readme-formatting) merged to main

## Key Findings

### Artifact Sync Process
- ISC artifacts must be synced via the Strong Compute control panel Experiments tab
- Artifacts sync to `/shared/artifacts/[experiment-name]/` on the remote workstation
- SCP transfer requires WireGuard VPN connection and container-specific SSH port
- Symlinks (`AtomicDirectory.latest_checkpoint`) cause SCP errors but don't affect data transfer

### Available Checkpoints for Analysis
14 experiments with checkpoints available locally:
1. exp-coherent-kind-character-251230 (checkpoint 84)
2. exp-cotton-working-xenon-251230 (checkpoint 27)
3. exp-elite-guttural-bubbler-251230 (checkpoint 4)
4. exp-future-plume-longship-251231 (checkpoint 6)
5. exp-glowing-rustic-giraffatitan-251230 (checkpoints 4, 55)
6. exp-heliotrope-leeward-fish-251229 (checkpoint 26)
7. exp-midnight-raspy-albertosaurus-251231 (checkpoint 80)
8. exp-organized-valley-fig-260101 (28 checkpoints - most complete training run)
9. exp-perfect-peaceful-travel-260101 (checkpoint 54)
10. exp-safe-tabby-dirigible-251229 (checkpoint 2)
11. exp-tremendous-gentle-passbook-251230 (checkpoint 23)

## Files Created/Modified

### New Files
- `ARTIFACT-SYNC.md` - Guide for syncing ISC artifacts to local machine

### Local Artifacts Synced
- `/Volumes/Extra FAT/Amanita-Validation/exp-organized-valley-fig-260101/checkpoints/` - 28 checkpoint directories
- `/Volumes/Extra FAT/Amanita-Validation/exp-safe-tabby-dirigible-251229/checkpoints/` - 1 checkpoint directory

## Environment Notes

### ISC Connection (Session-Specific)
- SSH: `ssh -p 52134 root@192.168.127.71`
- Requires WireGuard VPN connection
- Container ports change between sessions

### Local Artifact Storage
- Mac: `/Volumes/Extra FAT/Amanita-Validation/`
- Linux (previous session): `/media/j/Extra FAT/`

## Next Steps

1. **Launch TensorBoard** on synced experiments to analyze training curves
   ```bash
   tensorboard --logdir "/Volumes/Extra FAT/Amanita-Validation/"
   ```

2. **Compare checkpoints** across training runs to identify best performing model

3. **Validate multiple checkpoints** from exp-organized-valley-fig-260101 to find optimal epoch

4. **Investigate early stopping** - why did some experiments stop at low checkpoint numbers?

## Git Status at Handoff

```
On branch main
Your branch is up to date with 'origin/main'.

Untracked files:
  ARTIFACT-SYNC.md
```

ARTIFACT-SYNC.md needs to be committed.
