# Milestones

1) Run a training loop for a few examples. 

Use the instructions in this file `FungiTastic/baselines/closed_set/README.md` to run a training loop for just a few examples. The goal is to surface any issues with training in the local environment, not to improve model performance in any way. 

2) Modify architecture to output additional taxonomic ranks:

We will modify the following model so that it outputs not just the classified species, but also the ranks within parentheses below.

Kingdom > (Phylum > Class > Order > Family > Genus) > Species.

Since the dataset includes only fungi, we will omit kingdom. The existing architecture outputs species. The parenthesized taxonomic ranks will be added as classification outputs.

3) Modify loss function to account for the architectural changes above.

Since we now have six classification outputs, we must modify our training loss function to account for this fact. Suggest an appropriate loss function and implement it after review and approval. 
