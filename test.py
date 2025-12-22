import timm
import torch
import torchvision.transforms as T
from PIL import Image
from urllib.request import urlopen
import pandas as pd
import os

# Load model
model = timm.create_model("hf-hub:BVRA/beit_base_patch16_224.in1k_ft_fungitastic_224", pretrained=True)
model = model.eval()

# Image transforms
train_transforms = T.Compose([T.Resize((224, 224)), 
                              T.ToTensor(), 
                              T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]) 

# Load image and get prediction
img = Image.open('0-4100089130.JPG')
output = model(train_transforms(img).unsqueeze(0))

# Get predicted class index
predicted_class_idx = torch.argmax(output, dim=1).item()

print(f"Output shape: {output.shape}")
print(f"Output: {output}")
print(f"Probability (softmax) output: {torch.softmax(output, dim=1)}")
print(f"Predicted class index: {predicted_class_idx}")

# Map class index to species name
# Option 1: Load from training metadata CSV (if available)
# The metadata CSV should have 'class_id' and 'scientificName' columns
# You can download it from the FungiTastic dataset or use the dataset class
try:
    # Try to load from metadata CSV (adjust path as needed)
    # The metadata files are typically in: metadata/FungiTastic-Mini/FungiTastic-Mini-ClosedSet-Train.csv
    metadata_path = "FungiTastic/metadata/FungiTastic-Mini/FungiTastic-Mini-ClosedSet-Train.csv"
    if os.path.exists(metadata_path):
        train_df = pd.read_csv(metadata_path)
        # Create mapping from class_id to scientificName
        # Note: class_id in the model corresponds to category_id in the CSV
        id2species = dict(zip(train_df["category_id"], train_df["scientificName"]))
        # Ensure the mapping is sorted by class_id to match model output indices
        sorted_ids = sorted(id2species.keys())
        id2species_sorted = {i: id2species[sorted_ids[i]] for i in range(len(sorted_ids))}
        
        if predicted_class_idx < len(id2species_sorted):
            predicted_species = id2species_sorted[predicted_class_idx]
            print(f"Predicted species: {predicted_species}")
        else:
            print(f"Warning: Class index {predicted_class_idx} out of range")
    else:
        print(f"Metadata file not found at {metadata_path}")
        print("To get species names, you need to:")
        print("1. Download the FungiTastic dataset metadata")
        print("2. Load the training CSV file with 'category_id' and 'scientificName' columns")
        print("3. Create a mapping: id2species = dict(zip(df['category_id'], df['scientificName']))")
except Exception as e:
    print(f"Error loading metadata: {e}")
    print("To map class indices to species, you need the training metadata CSV file.")