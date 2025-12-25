# Milestones

1) Run a training loop for a few examples. 

Use the instructions in this file `FungiTastic/baselines/closed_set/README.md` to run a training loop for just a few examples. Ideally this retrains only the target model: 

`"hf-hub:BVRA/beit_base_patch16_224.in1k_ft_fungitastic_224"`

 The goal is to surface any issues with training and establish a working baseline for training the target model, not to improve model performance in any way. 

2) Modify architecture to output additional taxonomic ranks:

We will modify the following model so that it outputs not just the classified species, but also the ranks within parentheses below.

Kingdom > (Phylum > Class > Order > Family > Genus) > Species.

Since the dataset includes only fungi, we will omit kingdom. The existing architecture outputs species. The parenthesized taxonomic ranks will be added as classification outputs.

3) Modify loss function to account for the architectural changes above.

Since we now have six classification outputs, we must modify our training loss function to account for this fact. Suggest an appropriate loss function and implement it after review and approval. 

4) Modify training to be compatible with the Strong Compute ISC (cluster training):

Keep the existing locally-working training script intact, but create a new version, e.g.  train_isc_multitask.py which works on the Strong Compute ISC cluster.

For example, see docs here:
https://docs.strongcompute.com/getting-started/2.-hello-world-training-example#id-2.4-clone-the-strong-compute-isc-demos-github-repository

And the reference to cycling_utils, distributed training etc.

Also see examples here:
https://github.com/StrongResearch/isc-demos

This example:
https://github.com/StrongResearch/isc-demos/blob/main/fashion_mnist/train.py

In particular the cycling_utils usage:
```python
from cycling_utils import (
    InterruptableDistributedSampler,
    MetricsTracker,
    AtomicDirectory,
    atomic_torch_save,
)
```

and

```python
    ##############################################
    # Data Samplers and Loaders
    # ----------------------
    # Samplers for distributed training will typically assign a unique subset of the dataset to each GPU, shuffling
    # after each epoch, so that each GPU trains on a distinct subset of the dataset each epoch. The
    # InterruptibleDistributedSampler from cycling_utils by Strong Compute does this while also tracking progress of the
    # sampler through the dataset.

    train_sampler = InterruptableDistributedSampler(training_data)
    test_sampler = InterruptableDistributedSampler(test_data)
    timer.report("Initialized samplers")

    train_dataloader = DataLoader(training_data, batch_size=args.batch_size, sampler=train_sampler, num_workers=3)
    test_dataloader = DataLoader(test_data, batch_size=64, sampler=test_sampler)
    timer.report("Initialized dataloaders")
```

Also see references to:
```python
writer = SummaryWriter(log_dir=os.environ["LOSSY_ARTIFACT_PATH"])
```

And:
```python
    output_directory = os.environ["CHECKPOINT_ARTIFACT_PATH"]
    saver = AtomicDirectory(output_directory=output_directory, is_master=args.is_master)
```