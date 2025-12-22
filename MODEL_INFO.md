# Model Information and Usage Guide

## Model Architecture

The model used in this example is:
- **Architecture**: `beit_base_patch16_224` (BEiT: BERT Pre-Training of Image Transformers)
- **Source**: HuggingFace Hub model `BVRA/beit_base_patch16_224.in1k_ft_fungitastic_224`
- **Input Size**: 224x224 pixels
- **Framework**: PyTorch (via `timm` library)

The model definition comes from the `timm` (PyTorch Image Models) library. The architecture is a Vision Transformer (ViT) variant called BEiT.

## Model Definition

The model is created using:
```python
import timm
model = timm.create_model("hf-hub:BVRA/beit_base_patch16_224.in1k_ft_fungitastic_224", pretrained=True)
```

To see the full model architecture, you can:
```python
print(model)  # Print full model structure
```

The model architecture definition is in the `timm` library. You can inspect it:
```python
import timm
# Get model config
model_cfg = timm.models.get_pretrained_cfg("beit_base_patch16_224")
print(model_cfg)
```

## Training Script

The training script for this model is located at:
**`FungiTastic/baselines/closed_set/train.py`**

### Key Training Details:

1. **Training Configuration**: See `FungiTastic/baselines/closed_set/configs/FungiTastic-224.yaml`
2. **Model Loading**: Uses `load_model()` from `fgvc.utils.experiment` which creates models via `timm`
3. **Loss Functions**: Supports CrossEntropyLoss, FocalLoss, and SeeSawLoss
4. **Data Augmentation**: Uses "vit_heavy" augmentations
5. **Training Process**: 
   - Loads metadata CSV files with training/validation/test splits
   - Creates DataLoaders with appropriate transforms
   - Trains for specified epochs with validation
   - Saves best model based on F1 score

### To Run Training:

```bash
python FungiTastic/baselines/closed_set/train.py \
    --train-path <path_to_train_metadata.csv> \
    --val-path <path_to_val_metadata.csv> \
    --test-path <path_to_test_metadata.csv> \
    --config-path FungiTastic/baselines/closed_set/configs/FungiTastic-224.yaml \
    --wandb-entity <your_wandb_entity> \
    --wandb-project <your_wandb_project>
```

## Mapping Class Indices to Species Names

The model outputs logits for each class. The predicted class is the argmax of these logits. To map the class index back to the species name, you need the training metadata CSV file.

### Method 1: Using Training Metadata CSV

The training metadata CSV files contain `category_id` (which corresponds to the model's class indices) and `scientificName` columns.

```python
import pandas as pd

# Load training metadata
train_df = pd.read_csv("path/to/FungiTastic-Mini-ClosedSet-Train.csv")

# Create mapping from class_id to scientificName
# Note: The model uses sequential indices (0, 1, 2, ...) but the CSV uses category_id
# We need to create a sorted mapping
id2species = dict(zip(train_df["category_id"], train_df["scientificName"]))

# Sort by category_id to match model output indices
sorted_ids = sorted(id2species.keys())
id2species_sorted = {i: id2species[sorted_ids[i]] for i in range(len(sorted_ids))}

# Use the mapping
predicted_class_idx = torch.argmax(output, dim=1).item()
predicted_species = id2species_sorted[predicted_class_idx]
print(f"Predicted species: {predicted_species}")
```

### Method 2: Using FungiTastic Dataset Class

If you have the FungiTastic dataset installed, you can use the dataset class:

```python
from FungiTastic.dataset.fungi import FungiTastic

# Initialize dataset (this will load the metadata)
dataset = FungiTastic(
    root="path/to/dataset",
    data_subset="Mini",
    split="train",
    size="300",
    task="closed"
)

# The dataset has a category_id2label mapping
# But you need to map sequential indices (0, 1, 2, ...) to category_ids
# Get unique category_ids sorted
unique_category_ids = sorted(dataset.df["category_id"].unique())
category_id_to_idx = {cat_id: idx for idx, cat_id in enumerate(unique_category_ids)}
idx_to_category_id = {idx: cat_id for cat_id, idx in category_id_to_idx.items()}

# Map predicted index to species
predicted_class_idx = torch.argmax(output, dim=1).item()
category_id = idx_to_category_id[predicted_class_idx]
predicted_species = dataset.category_id2label[category_id]
print(f"Predicted species: {predicted_species}")
```

### Method 3: Direct Mapping from CSV (Simpler)

If the model was trained with sequential class indices (0, 1, 2, ...) matching sorted category_ids:

```python
import pandas as pd

# Load training metadata
train_df = pd.read_csv("path/to/FungiTastic-Mini-ClosedSet-Train.csv")

# Get unique category_ids and their corresponding species, sorted
unique_mapping = train_df.groupby("category_id")["scientificName"].first().sort_index()
id2species = unique_mapping.to_dict()

# Create sequential index mapping (assuming model uses 0, 1, 2, ...)
sorted_category_ids = sorted(id2species.keys())
idx2species = {i: id2species[cat_id] for i, cat_id in enumerate(sorted_category_ids)}

# Use the mapping
predicted_class_idx = torch.argmax(output, dim=1).item()
predicted_species = idx2species[predicted_class_idx]
print(f"Predicted species: {predicted_species}")
```

## Important Notes

1. **Class Index Alignment**: The model outputs indices 0, 1, 2, ..., N-1. These need to be mapped to the actual `category_id` values in the dataset, which may not be sequential.

2. **Metadata Location**: The metadata CSV files are typically located in:
   - `FungiTastic/metadata/FungiTastic-Mini/FungiTastic-Mini-ClosedSet-Train.csv`
   - `FungiTastic/metadata/FungiTastic-Mini/FungiTastic-Mini-ClosedSet-Val.csv`
   - `FungiTastic/metadata/FungiTastic-Mini/FungiTastic-Mini-ClosedSet-Test.csv`

3. **Column Names**: The CSV files use `category_id` (not `class_id`) and `scientificName` for species names.

4. **Model Output**: The model outputs raw logits. Apply softmax to get probabilities:
   ```python
   probabilities = torch.softmax(output, dim=1)
   ```

## Example: Complete Inference with Species Mapping

See the updated `test.py` file for a complete example that includes species name mapping.

