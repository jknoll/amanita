# Sync Artifacts from ISC Cluster

Instructions for syncing model checkpoints from the Strong Compute ISC cluster to local storage.

> **Note**: VPN is required to access the ISC cluster. If VPN is not available on the current host, use these instructions from a VPN-enabled machine.

## Remote Host Configuration

| Setting | Value |
|---------|-------|
| Host | `192.168.127.71` |
| Port | `52134` |
| User | `root` |
| Connection | `ssh -p 52134 root@192.168.127.71` |

## Paths

| Location | Path |
|----------|------|
| Remote checkpoints | `/shared/[exp-name]/checkpoints/` |
| Local destination | `/media/j/Extra FAT/Amanita-Validation/[exp-name]/` |

### Active Experiments

| Experiment | Status |
|------------|--------|
| `exp-tremendous-gentle-passbook-251230` | **Best performer** - only CP#23 synced |
| `exp-organized-valley-fig-260101` | 18 checkpoints synced |
| `exp-coherent-kind-character-251230` | CP#84 synced |
| `exp-perfect-peaceful-travel-260101` | CP#54 synced |

## Commands

### Test Connection

```bash
ssh -p 52134 root@192.168.127.71 "hostname && ls /shared/"
```

### List Available Checkpoints

```bash
# List all experiments
ssh -p 52134 root@192.168.127.71 "ls -la /shared/"

# List checkpoints for specific experiment
ssh -p 52134 root@192.168.127.71 "ls -la /shared/exp-tremendous-gentle-passbook-251230/checkpoints/"
```

### Sync All Checkpoints for an Experiment

```bash
# Sync entire checkpoints directory
rsync -avz --progress -e "ssh -p 52134" \
  root@192.168.127.71:/shared/exp-tremendous-gentle-passbook-251230/checkpoints/ \
  "/media/j/Extra FAT/Amanita-Validation/exp-tremendous-gentle-passbook-251230/checkpoints/"
```

### Sync Specific Checkpoint

```bash
# Sync a single checkpoint (e.g., checkpoint_20)
rsync -avz --progress -e "ssh -p 52134" \
  root@192.168.127.71:/shared/exp-tremendous-gentle-passbook-251230/checkpoints/AtomicDirectory_checkpoint_20/ \
  "/media/j/Extra FAT/Amanita-Validation/exp-tremendous-gentle-passbook-251230/checkpoints/AtomicDirectory_checkpoint_20/"
```

### Sync Multiple Specific Checkpoints

```bash
# Sync checkpoints 18-22 for bracket search
for cp in 18 19 20 21 22; do
  rsync -avz --progress -e "ssh -p 52134" \
    root@192.168.127.71:/shared/exp-tremendous-gentle-passbook-251230/checkpoints/AtomicDirectory_checkpoint_${cp}/ \
    "/media/j/Extra FAT/Amanita-Validation/exp-tremendous-gentle-passbook-251230/checkpoints/AtomicDirectory_checkpoint_${cp}/"
done
```

## Recommended Syncs (Bracket Search)

Based on checkpoint validation analysis, sync these additional checkpoints to refine the optimal early stopping point:

### High Priority

**`exp-tremendous-gentle-passbook-251230`** (best overall)
- Currently synced: CP#23 only
- Need to sync: CP#18, 19, 20, 21, 22 (to find true optimum before CP#23)
- Also consider: CP#24, 25, 26 (to confirm CP#23 is peak)

```bash
# Sync all checkpoints for best experiment
rsync -avz --progress -e "ssh -p 52134" \
  root@192.168.127.71:/shared/exp-tremendous-gentle-passbook-251230/checkpoints/ \
  "/media/j/Extra FAT/Amanita-Validation/exp-tremendous-gentle-passbook-251230/checkpoints/"
```

### Medium Priority

**`exp-coherent-kind-character-251230`** (second best)
- Currently synced: CP#84 only
- Consider syncing: Earlier checkpoints to track training progression

## After Syncing

After syncing new checkpoints, re-run validation:

```bash
cd /home/j/Documents/git/amanita-1
python validate_all_checkpoints.py
```

This will:
1. Scan for new checkpoints
2. Validate them on the FungiTastic validation set
3. Update the HTML report with new results
4. Identify the updated best checkpoint

## Troubleshooting

### Connection Refused

```bash
# Check if VPN is connected
ping 192.168.127.71

# If ping fails, connect to VPN first
```

### Permission Denied

```bash
# Ensure SSH key is added
ssh-add ~/.ssh/id_rsa

# Or specify key explicitly
rsync -avz --progress -e "ssh -p 52134 -i ~/.ssh/id_rsa" ...
```

### Disk Space

```bash
# Check available space on destination
df -h "/media/j/Extra FAT/"

# Each checkpoint is typically 500MB - 1GB
```

## Storage Locations

| Purpose | Path |
|---------|------|
| Synced checkpoints | `/media/j/Extra FAT/Amanita-Validation/` |
| Validation results | `/home/j/Documents/git/amanita-1/checkpoint_validation_results/` |
| Validation script | `/home/j/Documents/git/amanita-1/validate_all_checkpoints.py` |
