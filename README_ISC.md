# ISC Cluster Training Guide

This guide explains how to train the multi-task BEiT model on the Strong Compute ISC cluster.

## Overview

The `train_isc_multitask.py` script is specifically adapted for distributed training on the Strong Compute ISC cluster. It includes:

- **Distributed Training**: Multi-GPU training using PyTorch DDP
- **Interruptible Training**: Resume from checkpoints if interrupted
- **Distributed Sampling**: Each GPU processes a unique subset of data
- **Metrics Aggregation**: Properly aggregates metrics across GPUs
- **ISC Integration**: Uses cycling_utils for ISC-specific features

## Key Differences from Local Training

| Feature | Local (`train_multitask.py`) | ISC (`train_isc_multitask.py`) |
|---------|------------------------------|--------------------------------|
| Multi-GPU | Single GPU | Distributed (multi-GPU) |
| Sampling | Standard sampler | InterruptableDistributedSampler |
| Checkpointing | Standard torch.save | atomic_torch_save |
| Logging | Console only | TensorBoard + Console |
| Resumption | Manual | Automatic (via sampler state) |
| Metrics | Direct computation | Aggregated across GPUs |

## Prerequisites

1. **cycling_utils** package installed:
   ```bash
   pip install cycling-utils
   ```

2. **Dataset** available on ISC storage:
   - FungiTastic dataset at `/data/uds-fern-absorbed-dugong-251223/full_300px/`
   - Taxonomic mappings: `taxonomic_mappings.json`
   - Class weights: `class_weights.pt`

3. **ISC Environment Variables** (automatically set by ISC):
   - `CHECKPOINT_ARTIFACT_PATH`: Where to save model checkpoints
   - `LOSSY_ARTIFACT_PATH`: Where to save TensorBoard logs
   - `RANK`, `WORLD_SIZE`, `LOCAL_RANK`: Distributed training config

## Quick Start

### 0. Using the .isc Configuration File (Recommended)

The easiest way to launch training on ISC is using the `.isc` configuration file:

**File:** `fungitastic_multitask.isc`

```bash
# 1. Update the project ID in fungitastic_multitask.isc
#    Replace "<project-id>" with your actual ISC project ID

# 2. Launch the job using isc_run
isc_run fungitastic_multitask.isc

# 3. Monitor the job
isc status

# 4. View logs
isc logs <job-id>

# 5. Download checkpoints when complete
isc download <job-id>
```

**Configuration Details:**
- **Compute Mode:** `cycle` (uses preemptible instances for cost efficiency)
- **GPUs:** 4 (can be adjusted in the .isc file)
- **Dataset ID:** `uds-fern-absorbed-dugong-251223` (FungiTastic full dataset)
- **Batch Size:** 16 per GPU (effective batch size: 64)
- **Epochs:** 10
- **Class Balancing:** Enabled

To modify training parameters, edit the `command` section in `fungitastic_multitask.isc`.

### 1. Single GPU Test (Local)

Test the script locally on a single GPU:

```bash
python train_isc_multitask.py \
  --train-path ./minimal_subsets/FungiTastic-Train-Minimal.csv \
  --val-path ./minimal_subsets/FungiTastic-Val-Minimal.csv \
  --test-path ./minimal_subsets/FungiTastic-Test-Minimal.csv \
  --mappings-path ./taxonomic_mappings.json \
  --epochs 2 \
  --batch-size 8 \
  --is-master
```

### 2. Multi-GPU Training (using torchrun)

Train on 4 GPUs using PyTorch's distributed launcher:

```bash
torchrun \
  --nproc_per_node=4 \
  --nnodes=1 \
  train_isc_multitask.py \
  --train-path /data/uds-fern-absorbed-dugong-251223/full_300px/metadata/FungiTastic/FungiTastic-Train.csv \
  --val-path /data/uds-fern-absorbed-dugong-251223/full_300px/metadata/FungiTastic/FungiTastic-Val.csv \
  --test-path /data/uds-fern-absorbed-dugong-251223/full_300px/metadata/FungiTastic/FungiTastic-Test.csv \
  --mappings-path ./taxonomic_mappings.json \
  --class-weights-path ./class_weights.pt \
  --epochs 10 \
  --batch-size 16 \
  --lr 0.0001 \
  --use-class-weights
```

### 3. ISC Cluster Launch

Submit job to ISC cluster (if ISC CLI is available):

```bash
isc launch --config isc_config.yaml --gpus 4 --name fungitastic-run1
```

Or use the provided launch script:

```bash
./launch_isc.sh
```

## Configuration

### ISC File Configuration

The `fungitastic_multitask.isc` file contains the job configuration:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `isc_project_id` | `"<project-id>"` | Your ISC project ID (update this!) |
| `experiment_name` | `"fungitastic_multitask_beit"` | Name for this training run |
| `gpus` | `4` | Number of GPUs to use |
| `compute_mode` | `"cycle"` | Use preemptible instances (or "burst" for dedicated) |
| `dataset_id_list` | `["uds-fern-absorbed-dugong-251223"]` | FungiTastic dataset ID |
| `command` | Multi-line string | The training command to execute |

**Compute Modes:**
- `"cycle"`: Uses preemptible instances (cheaper, may be interrupted)
- `"burst"`: Uses dedicated instances (more expensive, guaranteed)

**Dataset ID:** `uds-fern-absorbed-dugong-251223`
- Contains the full FungiTastic dataset (433K train, 89K val, 91K test)
- Mounted at `/data/uds-fern-absorbed-dugong-251223/` in the container

### Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--train-path` | Required | Path to training CSV |
| `--val-path` | Required | Path to validation CSV |
| `--test-path` | Required | Path to test CSV |
| `--mappings-path` | `taxonomic_mappings.json` | Path to taxonomic mappings |
| `--class-weights-path` | `class_weights.pt` | Path to class weights |
| `--model-name` | `hf-hub:BVRA/...` | Base model identifier |
| `--pretrained` | `True` | Use pretrained weights |
| `--epochs` | `10` | Number of training epochs |
| `--batch-size` | `16` | Batch size per GPU |
| `--lr` | `0.0001` | Learning rate |
| `--workers` | `4` | Data loading workers |
| `--use-class-weights` | `False` | Enable class balancing |

### Distributed Arguments (auto-set by torchrun)

| Argument | Description |
|----------|-------------|
| `--local_rank` | GPU rank on current node |
| `--world-size` | Total number of GPUs |
| `--is-master` | Whether this is the master process |

## ISC-Specific Features

### 1. InterruptableDistributedSampler

Tracks progress through the dataset and can resume from interruptions:

```python
train_sampler = InterruptableDistributedSampler(train_dataset)
val_sampler = InterruptableDistributedSampler(val_dataset)
```

State is saved in checkpoints for seamless resumption.

### 2. MetricsTracker

Aggregates metrics across all GPUs:

```python
metrics_tracker = MetricsTracker(is_master=args.is_master)
aggregated_metrics = metrics_tracker.aggregate_metrics(local_metrics)
```

### 3. AtomicDirectory

Thread-safe checkpoint saving:

```python
saver = AtomicDirectory(
    output_directory=os.environ["CHECKPOINT_ARTIFACT_PATH"],
    is_master=args.is_master
)
atomic_torch_save(saver, checkpoint, 'checkpoint.pt')
```

### 4. TensorBoard Logging

Logs are saved to `LOSSY_ARTIFACT_PATH`:

```python
writer = SummaryWriter(log_dir=os.environ["LOSSY_ARTIFACT_PATH"])
writer.add_scalar('train/loss', loss, step)
```

## Monitoring Training

### TensorBoard

Launch TensorBoard to monitor training (if LOSSY_ARTIFACT_PATH is accessible):

```bash
tensorboard --logdir ${LOSSY_ARTIFACT_PATH}
```

### Checkpoints

Checkpoints are saved to `CHECKPOINT_ARTIFACT_PATH`:
- `checkpoint_epoch_N.pt`: Regular checkpoints
- `best_model.pt`: Best model based on validation F1

## Resuming Training

Training automatically resumes from the latest checkpoint if interrupted. The sampler state ensures you continue from the exact batch where you left off.

To manually resume from a specific checkpoint:

```python
checkpoint = torch.load('checkpoint_epoch_5.pt')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
train_sampler.load_state_dict(checkpoint['train_sampler_state'])
val_sampler.load_state_dict(checkpoint['val_sampler_state'])
```

## Performance Tips

### 1. Batch Size Tuning

The effective batch size is `batch_size * num_gpus`. For 4 GPUs with batch_size=16:
- Effective batch size: 64
- Total samples per step: 64

### 2. Learning Rate Scaling

When using more GPUs, consider scaling the learning rate:
- Rule of thumb: `lr_new = lr_base * sqrt(num_gpus)`
- Example: For 4 GPUs, `lr = 0.0001 * sqrt(4) = 0.0002`

### 3. Number of Workers

Set `--workers` based on available CPU cores:
- Recommended: `num_workers = 4 * num_gpus`
- For 4 GPUs: `--workers 16`

### 4. Mixed Precision Training

For faster training on A100/V100 GPUs, consider adding mixed precision:

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    outputs = model(images)
    loss = criterion(outputs, labels)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

## Troubleshooting

### Out of Memory Errors

Reduce batch size:
```bash
--batch-size 8  # Instead of 16
```

### Slow Data Loading

Increase workers:
```bash
--workers 8  # Instead of 4
```

### Gradient Synchronization Issues

Ensure all GPUs have the same number of batches by using drop_last:

```python
train_loader = DataLoader(..., drop_last=True)
```

### NCCL Errors

Set environment variables:
```bash
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1  # If InfiniBand causes issues
```

## Expected Performance

On 4x A100 GPUs (40GB) with batch_size=16:
- Training time: ~30-40 hours for 10 epochs (full dataset)
- Throughput: ~150-200 samples/sec
- Memory usage: ~25-30 GB per GPU

## References

- [Strong Compute ISC Documentation](https://docs.strongcompute.com/)
- [ISC Demo Repository](https://github.com/StrongResearch/isc-demos)
- [PyTorch Distributed Training](https://pytorch.org/tutorials/beginner/dist_overview.html)
- [cycling_utils Documentation](https://github.com/StrongResearch/cycling_utils)

## Support

For ISC-specific issues, contact Strong Compute support or refer to their documentation.
For model/training issues, see the main project README.md.
