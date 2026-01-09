import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from transformers import RobertaTokenizer
import clip
from typing import Dict, Tuple, Optional


class MultiModalDataset(Dataset):
    
    def __init__(
        self,
        data_path: str,
        image_dir: str,
        tokenizer: RobertaTokenizer,
        clip_preprocess,
        max_length: int = 512,
        mode: str = 'train'
    ):

        self.data = pd.read_csv(data_path)
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.clip_preprocess = clip_preprocess
        self.max_length = max_length
        self.mode = mode
        
        # Clean data: remove samples without text or images
        self._clean_data()
        
        print(f"Loaded {len(self.data)} samples for {mode} set")
        
    def _clean_data(self):
        # Remove samples without text
        self.data = self.data[self.data['text'].notna()]
        self.data = self.data[self.data['text'].str.len() > 10]
        
        # Verify image files exist
        if 'image_path' in self.data.columns:
            self.data['full_image_path'] = self.data['image_path'].apply(
                lambda x: os.path.join(self.image_dir, x) if pd.notna(x) else None
            )
            self.data['image_exists'] = self.data['full_image_path'].apply(
                lambda x: os.path.exists(x) if x else False
            )
            self.data = self.data[self.data['image_exists'] == True]
        
        # Reset index
        self.data = self.data.reset_index(drop=True)
        
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:

        row = self.data.iloc[idx]
        
        # Process text
        text = str(row['text'])
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Process image
        try:
            image_path = row['full_image_path']
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.clip_preprocess(image)
        except Exception as e:
            print(f"Error loading image at index {idx}: {e}")
            # Return a blank image if loading fails
            image_tensor = torch.zeros(3, 224, 224)
        
        # Get label
        label = int(row['label'])  # 0 for real, 1 for fake
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'image': image_tensor,
            'label': torch.tensor(label, dtype=torch.long),
            'text': text  # Keep original text for analysis
        }


def create_data_loaders(
    train_path: str,
    val_path: str,
    test_path: str,
    image_dir: str,
    batch_size: int = 32,
    max_length: int = 512,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, DataLoader]:

    # Initialize tokenizer and CLIP preprocessing
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    _, clip_preprocess = clip.load("ViT-B/32", device="cpu")
    
    # Create datasets
    train_dataset = MultiModalDataset(
        train_path, image_dir, tokenizer, clip_preprocess, max_length, 'train'
    )
    val_dataset = MultiModalDataset(
        val_path, image_dir, tokenizer, clip_preprocess, max_length, 'val'
    )
    test_dataset = MultiModalDataset(
        test_path, image_dir, tokenizer, clip_preprocess, max_length, 'test'
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def prepare_fakeddit_data(
    raw_data_path: str,
    output_dir: str,
    sample_size: Optional[int] = None,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15
):

    # Load data
    df = pd.read_csv(raw_data_path)
    
    # Filter for 2-way classification and samples with images
    if '2_way_label' in df.columns:
        df = df[df['2_way_label'].notna()]
        df['label'] = df['2_way_label']
    
    # Sample if requested
    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
    
    # Balance classes
    min_class_size = df['label'].value_counts().min()
    df = df.groupby('label').sample(n=min_class_size, random_state=42)
    
    # Create text column (combine title and body if available)
    if 'clean_title' in df.columns:
        df['text'] = df['clean_title']
        if 'selftext' in df.columns:
            df['text'] = df['text'] + ' ' + df['selftext'].fillna('')
    
    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Split data
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_df = df[:train_end]
    val_df = df[train_end:val_end]
    test_df = df[val_end:]
    
    # Save splits
    os.makedirs(output_dir, exist_ok=True)
    train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(output_dir, 'val.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)
    
    print(f"Data preparation complete:")
    print(f"  Train: {len(train_df)} samples")
    print(f"  Val: {len(val_df)} samples")
    print(f"  Test: {len(test_df)} samples")
    print(f"  Class distribution: {df['label'].value_counts().to_dict()}")


if __name__ == "__main__":
    
    print("To use this module:")
    print("1. Prepare your data using prepare_fakeddit_data()")
    print("2. Create data loaders using create_data_loaders()")
    print("3. Iterate through batches in your training loop")
