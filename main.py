import torch
import argparse
import os
from transformers import RobertaTokenizer

from data_loader_online import create_online_data_loaders
from model import MultiModalFakeNewsDetector, TextOnlyBaseline
from train import Trainer
from evaluate import ModelEvaluator


def main(args):
    print("MULTI-MODAL MISINFORMATION DETECTION")
    # Set device
    device = 'cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu'
    print(f"\nUsing device: {device}")
    
    # Step 1: Prepare data (if needed)
    if args.prepare_data:
        print("\n" + "=" * 70)
        print("STEP 1: Preparing Data")
        print("=" * 70)
        prepare_fakeddit_data(
            raw_data_path=args.raw_data_path,
            output_dir=args.data_dir,
            sample_size=args.sample_size,
            train_ratio=0.7,
            val_ratio=0.15
        )
    
    # Step 2: Create data loaders
    print("STEP 2: Loading Data")
    
    train_loader, val_loader, test_loader = create_online_data_loaders(
        train_path=os.path.join(args.data_dir, 'train.csv'),
        val_path=os.path.join(args.data_dir, 'val.csv'),
        test_path=os.path.join(args.data_dir, 'test.csv'),
        cache_dir='data/image_cache',
        batch_size=args.batch_size,
        max_length=args.max_length,
        num_workers=0  # Must be 0 for online downloading
    )
    
    print(f"✓ Training samples: {len(train_loader.dataset)}")
    print(f"✓ Validation samples: {len(val_loader.dataset)}")
    print(f"✓ Test samples: {len(test_loader.dataset)}")
    
    # Step 3: Initialize model
    print("STEP 3: Initializing Model")
    
    if args.model_type == 'multimodal':
        model = MultiModalFakeNewsDetector(
            roberta_model_name=args.roberta_model,
            num_classes=2,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout
        )
        print("✓ Multi-modal model initialized")
    elif args.model_type == 'text_only':
        model = TextOnlyBaseline(
            roberta_model_name=args.roberta_model,
            num_classes=2,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout
        )
        print("✓ Text-only baseline model initialized")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Step 4: Training
    if args.mode in ['train', 'train_eval']:
        print("STEP 4: Training Model")
        
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            learning_rate=args.learning_rate,
            num_epochs=args.num_epochs,
            warmup_steps=args.warmup_steps,
            output_dir=args.checkpoint_dir,
            patience=args.patience
        )
        
        trainer.train()
        
        # Load best model for evaluation
        best_checkpoint = torch.load(
            os.path.join(args.checkpoint_dir, 'best_model.pt'),
            map_location=device
        )
        model.load_state_dict(best_checkpoint['model_state_dict'])
        print(f"\n✓ Loaded best model (F1: {best_checkpoint['metrics']['f1']:.4f})")
    
    # Step 5: Evaluation
    if args.mode in ['eval', 'train_eval']:
        print("STEP 5: Evaluating Model")
        
        # Load model if only evaluating
        if args.mode == 'eval':
            checkpoint = torch.load(
                os.path.join(args.checkpoint_dir, 'best_model.pt'),
                map_location=device
            )
            model.load_state_dict(checkpoint['model_state_dict'])
            print("✓ Loaded model from checkpoint")
        
        evaluator = ModelEvaluator(
            model=model,
            test_loader=test_loader,
            device=device,
            output_dir=args.results_dir
        )
        
        # Run evaluation
        metrics = evaluator.evaluate()
        
        # Error analysis
        if args.error_analysis:
            evaluator.error_analysis(num_examples=args.num_error_examples)
        
        # Attention visualization (example)
        if args.visualize_attention:
            tokenizer = RobertaTokenizer.from_pretrained(args.roberta_model)
            example_text = test_loader.dataset[0]['text']
            print(f"\nVisualizing attention for example text:")
            print(f"{example_text[:200]}...")
            evaluator.visualize_attention(example_text, tokenizer)
    
    print("PIPELINE COMPLETE!")
    print(f"\nResults saved to: {args.results_dir}")
    print(f"Checkpoints saved to: {args.checkpoint_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description='Multi-Modal Misinformation Detection'
    )
    
    # Mode
    parser.add_argument(
        '--mode',
        type=str,
        default='train_eval',
        choices=['train', 'eval', 'train_eval'],
        help='Execution mode: train, eval, or train_eval'
    )
    
    # Data arguments
    parser.add_argument('--prepare_data', action='store_true',
                       help='Prepare data from raw files')
    parser.add_argument('--raw_data_path', type=str, default='./data/raw/fakeddit.csv',
                       help='Path to raw Fakeddit CSV')
    parser.add_argument('--data_dir', type=str, default='./data/processed',
                       help='Directory for processed data')
    parser.add_argument('--image_dir', type=str, default='./data/images',
                       help='Directory containing images')
    parser.add_argument('--sample_size', type=int, default=None,
                       help='Number of samples to use (None for all)')
    
    # Model arguments
    parser.add_argument('--model_type', type=str, default='multimodal',
                       choices=['multimodal', 'text_only'],
                       help='Type of model to use')
    parser.add_argument('--roberta_model', type=str, default='roberta-base',
                       help='Pre-trained RoBERTa model name')
    parser.add_argument('--hidden_dim', type=int, default=256,
                       help='Hidden dimension for classifier')
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout probability')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                       help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=5,
                       help='Number of training epochs')
    parser.add_argument('--warmup_steps', type=int, default=0,
                       help='Number of warmup steps')
    parser.add_argument('--patience', type=int, default=3,
                       help='Early stopping patience')
    parser.add_argument('--max_length', type=int, default=512,
                       help='Maximum sequence length')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    
    # Evaluation arguments
    parser.add_argument('--error_analysis', action='store_true',
                       help='Perform error analysis')
    parser.add_argument('--num_error_examples', type=int, default=50,
                       help='Number of error examples to analyze')
    parser.add_argument('--visualize_attention', action='store_true',
                       help='Visualize attention weights')
    
    # Output arguments
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                       help='Directory to save model checkpoints')
    parser.add_argument('--results_dir', type=str, default='./results',
                       help='Directory to save evaluation results')
    
    # Device arguments
    parser.add_argument('--no_cuda', action='store_true',
                       help='Disable CUDA even if available')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
