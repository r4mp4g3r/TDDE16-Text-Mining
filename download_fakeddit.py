# Save as download_fakeddit.py
from datasets import load_dataset
import os

print("Downloading Fakeddit dataset...")
print()

# Download the 2-way classification version (Real vs Fake)
dataset = load_dataset("fakeddit", "2way")

print(f"✓ Downloaded!")
print(f"  Train: {len(dataset['train'])} samples")
print(f"  Validation: {len(dataset['validation'])} samples") 
print(f"  Test: {len(dataset['test'])} samples")

# Create a manageable subset (50K samples)
print("\nCreating subset of 50,000 samples...")

train_subset = dataset['train'].shuffle(seed=42).select(range(35000))
val_subset = dataset['validation'].shuffle(seed=42).select(range(7500))
test_subset = dataset['test'].shuffle(seed=42).select(range(7500))

# Save to disk
os.makedirs('../data/processed', exist_ok=True)
os.makedirs('../data/images', exist_ok=True)

print("Saving data...")
train_subset.to_csv('../data/processed/train.csv', index=False)
val_subset.to_csv('../data/processed/val.csv', index=False)
test_subset.to_csv('../data/processed/test.csv', index=False)

# Save images
print("Saving images (this may take a while)...")
for split_name, split_data in [('train', train_subset), ('val', val_subset), ('test', test_subset)]:
    for i, example in enumerate(split_data):
        if example['image'] is not None:
            img_path = f"../data/images/{split_name}_{i}.jpg"
            example['image'].save(img_path)
        if i % 1000 == 0:
            print(f"  {split_name}: {i}/{len(split_data)} images saved")

print("\n✅ Dataset ready!")
print(f"  Location: ../data/processed/")
print(f"  Images: ../data/images/")
