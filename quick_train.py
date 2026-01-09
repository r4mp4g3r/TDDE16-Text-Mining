import torch
import argparse
import os
import pandas as pd

from data_loader_online import create_online_data_loaders
from model import MultiModalFakeNewsDetector
from train import Trainer
from evaluate import ModelEvaluator


def create_small_dataset(input_path, output_path, size=5000):
    df = pd.read_csv(input_path)
    
    # Balance classes
    if 'label' in df.columns:
        df_0 = df[df['label'] == 0].sample(n=min(len(df[df['label'] == 0]), size//2), random_state=42)
        df_1 = df[df['label'] == 1].sample(n=min(len(df[df['label'] == 1]), size//2), random_state=42)
        df_small = pd.concat([df_0, df_1]).sample(frac=1, random_state=42)
    else:
        df_small = df.sample(n=min(len(df), size), random_state=42)
    
    df_small.to_csv(output_path, index=False)
    return len(df_small)


def quick_train(
    data_dir='data/processed',
    small_data_dir='data/small',
    train_size=5000,
    val_size=1000,
    test_size=1000,
    batch_size=32,
    num_epochs=3,
    use_mps=True
):
    
    print(f"\nOptimizations:")
    print(f"  - Smaller dataset: {train_size + val_size + test_size} samples")
    print(f"  - Larger batch size: {batch_size}")
    print(f"  - Fewer epochs: {num_epochs}")
    print(f"  - Expected time: 30-60 minutes")
    
    # Create smaller dataset
    print("CREATING SMALLER DATASET")
    
    os.makedirs(small_data_dir, exist_ok=True)
    
    train_count = create_small_dataset(
        os.path.join(data_dir, 'train.csv'),
        os.path.join(small_data_dir, 'train.csv'),
        train_size
    )
    val_count = create_small_dataset(
        os.path.join(data_dir, 'val.csv'),
        os.path.join(small_data_dir, 'val.csv'),
        val_size
    )
    test_count = create_small_dataset(
        os.path.join(data_dir, 'test.csv'),
        os.path.join(small_data_dir, 'test.csv'),
        test_size
    )
    
    print(f"✓ Created smaller dataset:")
    print(f"  Train: {train_count} samples")
    print(f"  Val: {val_count} samples")
    print(f"  Test: {test_count} samples")
    
    # Set device
    if use_mps and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
        print(f"\n✓ Using Apple Silicon GPU (MPS)")
    elif torch.cuda.is_available():
        device = 'cuda'
        print(f"\n✓ Using CUDA GPU")
    else:
        device = 'cpu'
        print(f"\n⚠ Using CPU (will be slow)")
    
    # Load data
    print(f"\n{'='*70}")
    print("LOADING DATA")
    print("="*70)
    
    train_loader, val_loader, test_loader = create_online_data_loaders(
        train_path=os.path.join(small_data_dir, 'train.csv'),
        val_path=os.path.join(small_data_dir, 'val.csv'),
        test_path=os.path.join(small_data_dir, 'test.csv'),
        cache_dir='data/image_cache_small',
        batch_size=batch_size,
        num_workers=0
    )
    
    print(f"✓ Data loaded")
    
    # Initialize model
    print("INITIALIZING MODEL")
    
    model = MultiModalFakeNewsDetector(
        roberta_model_name='roberta-base',
        num_classes=2,
        hidden_dim=256,
        dropout=0.3
    )
    
    print(f"✓ Model initialized")
    
    # Train
    print("TRAINING")
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=2e-5,
        num_epochs=num_epochs,
        output_dir='./checkpoints_quick',
        patience=2
    )
    
    trainer.train()
    
    # Evaluate
    print("EVALUATION")
    
    best_checkpoint = torch.load(
        os.path.join('./checkpoints_quick', 'best_model.pt'),
        map_location=device
    )
    model.load_state_dict(best_checkpoint['model_state_dict'])
    
    evaluator = ModelEvaluator(
        model=model,
        test_loader=test_loader,
        device=device,
        output_dir='./results_quick'
    )
    
    metrics = evaluator.evaluate()
    
    print("QUICK TRAINING COMPLETE!")
    print(f"\nResults:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  F1-Score: {metrics['f1']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    
    print(f"\n📋 Next Steps:")
    print(f"  1. Results saved to: ./results_quick/")
    print(f"  2. Model saved to: ./checkpoints_quick/")
    print(f"  3. For full training, use Google Colab (see instructions)")
    
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Quick training mode')
    parser.add_argument('--train_size', type=int, default=5000,
                       help='Number of training samples')
    parser.add_argument('--val_size', type=int, default=1000,
                       help='Number of validation samples')
    parser.add_argument('--test_size', type=int, default=1000,
                       help='Number of test samples')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=3,
                       help='Number of epochs')
    parser.add_argument('--no_mps', action='store_true',
                       help='Disable MPS (Apple Silicon GPU)')
    
    args = parser.parse_args()
    
    quick_train(
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        use_mps=not args.no_mps
    )
