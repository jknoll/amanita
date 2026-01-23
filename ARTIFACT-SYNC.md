# Artifact Sync Guide

This guide explains how to sync experiment artifacts (logs, checkpoints, TensorBoard data) from Strong Compute ISC to your local machine.

## Overview

Artifacts from ISC training runs include:
- **System console logs**: stdout/stderr from the training process
- **TensorBoard logs**: Training metrics, loss curves, validation results
- **Checkpoints**: Model weights, optimizer state, sampler state

## Prerequisites

1. **WireGuard VPN**: Must be connected to the Strong Compute network. See [Strong Compute VPN documentation](https://docs.strongcompute.com/) for setup.

2. **SSH keys**: Your SSH public key must be deployed on the ISC server.

3. **Running container**: You must have an active container session on the ISC cluster.

## Step 1: Start a Container

Launch a container on the ISC cluster through the Strong Compute control panel or CLI.

## Step 2: Trigger Artifact Sync

1. Navigate to the **Experiments** tab in the Strong Compute control panel
2. Select the experiment(s) you want to sync
3. Choose which artifacts to sync:
   - System console logs
   - TensorBoard logs
   - Checkpoints
4. Initiate the sync

Artifacts will be synced to the remote workstation at:
```
/shared/artifacts/[experiment-name]/
```

## Step 3: Copy to Local Machine

Use SCP to copy artifacts from the remote workstation to your local machine.

### Get SSH Command

The SSH command is container-specific. Obtain it from the Strong Compute control panel for your active container session.

Example format:
```bash
ssh -p <port> user@<host>
```

### SCP Commands

Copy a specific experiment's artifacts:
```bash
scp -r -P <port> user@<host>:/shared/artifacts/<experiment-name>/ ./artifacts/
```

Copy all artifacts:
```bash
scp -r -P <port> user@<host>:/shared/artifacts/ ./artifacts/
```

Copy only TensorBoard logs:
```bash
scp -r -P <port> user@<host>:/shared/artifacts/<experiment-name>/tensorboard/ ./artifacts/<experiment-name>/tensorboard/
```

Copy only checkpoints:
```bash
scp -r -P <port> user@<host>:/shared/artifacts/<experiment-name>/checkpoints/ ./artifacts/<experiment-name>/checkpoints/
```

## Step 4: Analyze Artifacts

### View TensorBoard Logs

```bash
tensorboard --logdir ./artifacts/<experiment-name>/tensorboard/
```

Or use the provided launch script for multiple experiments:
```bash
./validation/launch_tensorboard.sh
```

### Load Checkpoints

```python
import torch

checkpoint = torch.load('./artifacts/<experiment-name>/checkpoints/best_model.pt', weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
```

## Directory Structure

After syncing, your local artifacts directory should look like:
```
artifacts/
├── exp-experiment-name-1/
│   ├── checkpoints/
│   │   ├── checkpoint_epoch_1.pt
│   │   ├── checkpoint_epoch_2.pt
│   │   └── best_model.pt
│   ├── tensorboard/
│   │   └── events.out.tfevents.*
│   └── logs/
│       └── console.log
├── exp-experiment-name-2/
│   └── ...
```

## Troubleshooting

### Connection Refused
- Verify WireGuard VPN is connected
- Check that the container is still running
- Confirm the SSH port is correct

### Permission Denied
- Ensure SSH keys are properly deployed
- Check file permissions on the remote server

### Sync Not Appearing
- Wait a few minutes for the sync to complete
- Check the Experiments tab for sync status
- Verify the experiment name matches the path

## References

- [Strong Compute Documentation](https://docs.strongcompute.com/)
- [README_ISC.md](./README_ISC.md) - ISC training guide
