import pandas as pd
import os
from pathlib import Path
import requests
from PIL import Image
from io import BytesIO
import time

def setup_fakeddit_from_kaggle(
    raw_data_dir: str = '../data/raw',
    output_dir: str = '../data/processed',
    sample_size: int = 50000,
    test_download: bool = True
):
    
    print("FAKEDDIT DATASET SETUP")
    
    # Check if TSV files exist
    train_path = os.path.join(raw_data_dir, 'multimodal_train.tsv')
    val_path = os.path.join(raw_data_dir, 'multimodal_validate.tsv')
    test_path = os.path.join(raw_data_dir, 'multimodal_test_public.tsv')
    
    if not os.path.exists(train_path):
        print(f"\n❌ Error: Could not find {train_path}")
        return
    
    print(f"\n✓ Found TSV files in {raw_data_dir}")
    
    # Load data
    print("\nLoading data (this may take a minute)...")
    train_df = pd.read_csv(train_path, sep='\t', low_memory=False)
    val_df = pd.read_csv(val_path, sep='\t', low_memory=False)
    test_df = pd.read_csv(test_path, sep='\t', low_memory=False)
    
    print(f"✓ Loaded data:")
    print(f"  Train: {len(train_df)} samples")
    print(f"  Validation: {len(val_df)} samples")
    print(f"  Test: {len(test_df)} samples")
    
    # Explore columns
    print(f"\nColumns in dataset: {list(train_df.columns)[:10]}...")
    
    # Filter for samples with images and 2-way classification
    print("\nFiltering for multi-modal samples...")
    
    # Check which columns exist
    has_image_col = 'hasImage' if 'hasImage' in train_df.columns else 'has_image'
    label_col = '2_way_label' if '2_way_label' in train_df.columns else 'label'
    
    train_df = train_df[train_df[has_image_col] == True]
    val_df = val_df[val_df[has_image_col] == True]
    test_df = test_df[test_df[has_image_col] == True]
    
    print(f"✓ After filtering:")
    print(f"  Train: {len(train_df)} samples with images")
    print(f"  Validation: {len(val_df)} samples with images")
    print(f"  Test: {len(test_df)} samples with images")
    
    # Create balanced subset
    print(f"\nCreating balanced subset of {sample_size} samples...")
    
    # Calculate split sizes
    train_size = int(sample_size * 0.7)
    val_size = int(sample_size * 0.15)
    test_size = int(sample_size * 0.15)
    
    # Balance classes
    def balance_and_sample(df, size):
        if label_col in df.columns:
            # Group by label and sample equally
            min_class = df[label_col].value_counts().min()
            balanced = df.groupby(label_col).sample(
                n=min(min_class, size // 2),
                random_state=42
            )
            return balanced.sample(n=min(len(balanced), size), random_state=42)
        else:
            return df.sample(n=min(len(df), size), random_state=42)
    
    train_subset = balance_and_sample(train_df, train_size)
    val_subset = balance_and_sample(val_df, val_size)
    test_subset = balance_and_sample(test_df, test_size)
    
    print(f"✓ Created subsets:")
    print(f"  Train: {len(train_subset)} samples")
    print(f"  Validation: {len(val_subset)} samples")
    print(f"  Test: {len(test_subset)} samples")
    
    # Prepare data for our model
    print("\nPreparing data for model...")
    
    def prepare_dataframe(df):
        prepared = pd.DataFrame()
        
        # ID
        prepared['id'] = df['id'] if 'id' in df.columns else df.index
        
        # Text (combine title and body if available)
        if 'clean_title' in df.columns:
            prepared['text'] = df['clean_title'].fillna('')
            if 'selftext' in df.columns:
                prepared['text'] = prepared['text'] + ' ' + df['selftext'].fillna('')
        elif 'title' in df.columns:
            prepared['text'] = df['title'].fillna('')
        else:
            prepared['text'] = ''
        
        # Image URL
        prepared['image_url'] = df['image_url'] if 'image_url' in df.columns else ''
        
        # Label (0 = real, 1 = fake)
        if label_col in df.columns:
            prepared['label'] = df[label_col].astype(int)
        else:
            prepared['label'] = 0  # Default if no label
        
        # Keep only rows with text
        prepared = prepared[prepared['text'].str.len() > 10]
        
        return prepared
    
    train_prepared = prepare_dataframe(train_subset)
    val_prepared = prepare_dataframe(val_subset)
    test_prepared = prepare_dataframe(test_subset)
    
    # Save processed data
    os.makedirs(output_dir, exist_ok=True)
    
    train_prepared.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    val_prepared.to_csv(os.path.join(output_dir, 'val.csv'), index=False)
    test_prepared.to_csv(os.path.join(output_dir, 'test.csv'), index=False)
    
    print(f"\n✓ Saved processed data to {output_dir}")
    
    # Test downloading a few images
    if test_download:
        print("\nTesting image download (downloading 5 sample images)...")
        test_image_download(train_prepared, output_dir)
    
    # Print statistics
    print("SETUP COMPLETE!")
    print(f"\nDataset Statistics:")
    print(f"  Total samples: {len(train_prepared) + len(val_prepared) + len(test_prepared)}")
    print(f"  Train: {len(train_prepared)} samples")
    print(f"  Validation: {len(val_prepared)} samples")
    print(f"  Test: {len(test_prepared)} samples")
    
    if label_col in train_prepared.columns:
        print(f"\nClass distribution (train):")
        print(train_prepared['label'].value_counts())
    
    print(f"\nData saved to:")
    print(f"  {os.path.abspath(output_dir)}")
    
    print(f"\n📋 Next Steps:")
    print(f"  1. Images will be downloaded automatically during training")
    print(f"  2. Run: python main.py --mode train_eval --batch_size 16")
    print(f"  3. Training will start immediately!")
    
    return train_prepared, val_prepared, test_prepared


def test_image_download(df, output_dir, num_test=5):
    
    image_dir = os.path.join(output_dir, '../images/test')
    os.makedirs(image_dir, exist_ok=True)
    
    success_count = 0
    
    for idx, row in df.head(num_test).iterrows():
        try:
            url = row['image_url']
            if pd.isna(url) or url == '':
                continue
            
            # Download image
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                img = Image.open(BytesIO(response.content))
                img = img.convert('RGB')
                
                # Save test image
                img_path = os.path.join(image_dir, f'test_{idx}.jpg')
                img.save(img_path)
                
                success_count += 1
                print(f"  ✓ Downloaded test image {success_count}/{num_test}")
            
            time.sleep(0.5)  # Be respectful to servers
            
        except Exception as e:
            print(f"  ⚠ Failed to download image {idx}: {str(e)[:50]}")
    
    if success_count > 0:
        print(f"\n✓ Successfully downloaded {success_count}/{num_test} test images")
        print(f"  Images saved to: {image_dir}")
    else:
        print(f"\n⚠ Warning: Could not download test images")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Setup Fakeddit dataset from Kaggle')
    parser.add_argument('--raw_dir', type=str, default='../data/raw',
                       help='Directory containing downloaded TSV files')
    parser.add_argument('--output_dir', type=str, default='../data/processed',
                       help='Directory to save processed data')
    parser.add_argument('--sample_size', type=int, default=50000,
                       help='Number of samples to use')
    parser.add_argument('--no_test', action='store_true',
                       help='Skip testing image downloads')
    
    args = parser.parse_args()
    
    setup_fakeddit_from_kaggle(
        raw_data_dir=args.raw_dir,
        output_dir=args.output_dir,
        sample_size=args.sample_size,
        test_download=not args.no_test
    )
