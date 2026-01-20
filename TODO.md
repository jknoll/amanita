1) 
The fungitastic dataset is locally mounted at: /media/j/Extra\ FAT/FungiTastic/dataset/FungiTastic/.
The last multitask training checkpoint is in /artifacts/[experiment_name]

Create a jupyter notebook which performs validation on the trained multitask model using the validation dataset. Calculate precision and accuracy metrics at
each taxonomic rank, calculate and render a confusion matrix for each rank, and inline render some images of specimens in the dataset which were well recognized as well as those which were not. Provide notebook code examples of how to validate on particular observations.

The notebook should include in its output how "confident" the model is in the classification at each taxonomic rank.

Pay particular attention to cases like the Amanita Phalloides, or Death Cap. Are we able to accurately recognize it to the species level? If not, are we at least able to recognize it as an Amanita with high confidence?

Render an html report with the results.

