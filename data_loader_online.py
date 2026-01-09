import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from transformers import RobertaTokenizer
import clip
from typing import Dict, Tuple, Optional
import requests
from io import BytesIO
import time
from pathlib import Path


class OnlineMultiModalDataset(Dataset):
    
    def __init__(
        self,
        data_path: str,
        cache_dir: str,
        tokenizer: RobertaTokenizer,
        clip_preprocess,
        max_length: int = 512,
        mode: str = 'train',
        max_retries: int = 2
    ):

        self.data = pd.read_csv(data_path)
        self.cache_dir = cache_dir
        self.tokenizer = tokenizer
        self.clip_preprocess = clip_preprocess
        self.max_length = max_length
        self.mode = mode
        self.max_retries = max_retries
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
        
        # Clean data
        self._clean_data()
        
        print(f"Loaded {len(self.data)} samples for {mode} set")
        print(f"Images will be downloaded on-the-fly and cached to {cache_dir}")
        
    def _clean_data(self):
        """Remove samples with missing text or image URLs"""
        # Remove samples without text
        self.data = self.data[self.data['text'].notna()]
        self.data = self.data[self.data['text'].str.len() > 10]
        
        # Remove samples without image URLs
        self.data = self.data[self.data['image_url'].notna()]
        self.data = self.data[self.data['image_url'] != '']
        
        # Reset index
        self.data = self.data.reset_index(drop=True)
        
    def _get_cache_path(self, image_id: str) -> str:
        """Get cache file path for an image"""
        return os.path.join(self.cache_dir, f"{image_id}.jpg")
    
    def _download_image(self, url: str, image_id: str) -> Optional[Image.Image]:
        cache_path = self._get_cache_path(image_id)
        
        # Check if already cached
        if os.path.exists(cache_path):
            try:
                return Image.open(cache_path).convert('RGB')
            except:
                # If cached file is corrupted, delete it
                os.remove(cache_path)
        
        # Download image
        for attempt in range(self.max_retries):
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    img = Image.open(BytesIO(response.content))
                    img = img.convert('RGB')
                    
                    # Cache the image
                    img.save(cache_path)
                    
                    return img
                
            except Exception as e:
                if attempt == self.max_retries - 1:
                    # Last attempt failed
                    return None
                time.sleep(0.5)  # Wait before retry
        
        return None
    
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
        
        # Download and process image
        image_id = str(row['id'])
        image_url = str(row['image_url'])
        
        image = self._download_image(image_url, image_id)
        
        if image is not None:
            try:
                image_tensor = self.clip_preprocess(image)
            except:
                # If preprocessing fails, use blank image
                image_tensor = torch.zeros(3, 224, 224)
        else:
            # If download fails, use blank image
            image_tensor = torch.zeros(3, 224, 224)
        
        # Get label
        label = int(row['label']) if 'label' in row else 0
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'image': image_tensor,
            'label': torch.tensor(label, dtype=torch.long),
            'text': text
        }


def create_online_data_loaders(
    train_path: str,
    val_path: str,
    test_path: str,
    cache_dir: str = '../data/image_cache',
    batch_size: int = 32,
    max_length: int = 512,
    num_workers: int = 0  # Set to 0 for online downloading
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    # Initialize tokenizer and CLIP preprocessing
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    _, clip_preprocess = clip.load("ViT-B/32", device="cpu")
    
    # Create cache directories
    train_cache = os.path.join(cache_dir, 'train')
    val_cache = os.path.join(cache_dir, 'val')
    test_cache = os.path.join(cache_dir, 'test')
    
    # Create datasets
    train_dataset = OnlineMultiModalDataset(
        train_path, train_cache, tokenizer, clip_preprocess, max_length, 'train'
    )
    val_dataset = OnlineMultiModalDataset(
        val_path, val_cache, tokenizer, clip_preprocess, max_length, 'val'
    )
    test_dataset = OnlineMultiModalDataset(
        test_path, test_cache, tokenizer, clip_preprocess, max_length, 'test'
    )
    
    # Create data loaders
    # Note: num_workers=0 is recommended for online downloading to avoid issues
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


if __name__ == "__main__":
    print("Online Multi-Modal Data Loader")
    
    print("\nThis data loader downloads images on-the-fly from URLs.")
    print("Images are cached locally to avoid re-downloading.")
    
    print("\nExample usage:")
    print("""
    from data_loader_online import create_online_data_loaders
    
    train_loader, val_loader, test_loader = create_online_data_loaders(
        train_path='../data/processed/train.csv',
        val_path='../data/processed/val.csv',
        test_path='../data/processed/test.csv',
        cache_dir='../data/image_cache',
        batch_size=16
    )
    """)
