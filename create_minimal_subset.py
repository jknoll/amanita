import pandas as pd
import os

# Paths
data_path = "/data/uds-fern-absorbed-dugong-251223/full_300px/metadata/FungiTastic"
output_path = "./minimal_subsets"

os.makedirs(output_path, exist_ok=True)

print("Loading full datasets...")
# Load full datasets
train_df = pd.read_csv(f"{data_path}/FungiTastic-Train.csv")
val_df = pd.read_csv(f"{data_path}/FungiTastic-ClosedSet-Val.csv")
test_df = pd.read_csv(f"{data_path}/FungiTastic-ClosedSet-Test.csv")

print(f"Original sizes - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

# Strategy: Sample proportionally from each class to maintain distribution
# Get top N most frequent classes
n_classes = 50  # Use subset of classes
n_samples_per_class_train = 10  # 10 samples per class in train
n_samples_per_class_val = 2  # 2 samples per class in val
n_samples_per_class_test = 2  # 2 samples per class in test

print(f"\nSelecting top {n_classes} most frequent classes...")
# Get most frequent classes from training set
top_classes = train_df['category_id'].value_counts().head(n_classes).index.tolist()

print(f"Sampling {n_samples_per_class_train} training samples per class...")
# Filter and sample
train_subset = train_df[train_df['category_id'].isin(top_classes)].groupby('category_id').head(n_samples_per_class_train).copy()
val_subset = val_df[val_df['category_id'].isin(top_classes)].groupby('category_id').head(n_samples_per_class_val).copy()
test_subset = test_df[test_df['category_id'].isin(top_classes)].groupby('category_id').head(n_samples_per_class_test).copy()

# Add required columns for FGVC training
# 1. Rename category_id to class_id
train_subset['class_id'] = train_subset['category_id']
val_subset['class_id'] = val_subset['category_id']
test_subset['class_id'] = test_subset['category_id']

# 2. Create image_path from filename
data_root = "/data/uds-fern-absorbed-dugong-251223/full_300px/FungiTastic"
train_subset['image_path'] = train_subset['filename'].apply(lambda x: f"{data_root}/train/300p/{x}")
val_subset['image_path'] = val_subset['filename'].apply(lambda x: f"{data_root}/val/300p/{x}")
test_subset['image_path'] = test_subset['filename'].apply(lambda x: f"{data_root}/test/300p/{x}")

print(f"\nSubset statistics:")
print(f"Train subset: {len(train_subset)} samples, {train_subset['category_id'].nunique()} classes")
print(f"Val subset: {len(val_subset)} samples, {val_subset['category_id'].nunique()} classes")
print(f"Test subset: {len(test_subset)} samples, {test_subset['category_id'].nunique()} classes")

# Verify image paths exist for a few samples
print(f"\nVerifying image paths...")
sample_paths = train_subset['image_path'].head(3).tolist()
for path in sample_paths:
    exists = os.path.exists(path)
    filename = os.path.basename(path)
    print(f"  {filename}: {'✓' if exists else '✗'}")

# Save subsets
print(f"\nSaving subsets to {output_path}/...")
train_subset.to_csv(f"{output_path}/FungiTastic-Train-Minimal.csv", index=False)
val_subset.to_csv(f"{output_path}/FungiTastic-Val-Minimal.csv", index=False)
test_subset.to_csv(f"{output_path}/FungiTastic-Test-Minimal.csv", index=False)

print(f"✓ Subsets saved successfully!")
