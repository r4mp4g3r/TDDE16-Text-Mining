import torch
import numpy as np
import pandas as pd
import json
import os
from sklearn.metrics import confusion_matrix

# Fix PyTorch 2.6
try:
    torch.serialization.add_safe_globals([np._core.multiarray.scalar])
except:
    torch.serialization.add_safe_globals([np.core.multiarray.scalar])

def analyze_errors(model, test_loader, device='cuda', num_examples=50):
    print("Analyzing errors...")
    model.eval()
    
    false_positives = []  # Real classified as Fake
    false_negatives = []  # Fake classified as Real
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            texts = batch.get('text', [''] * len(labels))
            
            batch_size = input_ids.shape[0]
            linguistic_features = torch.zeros(batch_size, 11).to(device)
            
            output = model(input_ids, attention_mask, images, linguistic_features)
            
            if isinstance(output, dict):
                logits = output.get('logits', list(output.values())[0])
            else:
                logits = output
            
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            # Find errors
            for i in range(len(labels)):
                pred = preds[i].item()
                true = labels[i].item()
                confidence = probs[i, pred].item()
                text = texts[i] if isinstance(texts, list) else texts[i].item() if hasattr(texts[i], 'item') else str(texts[i])
                
                if pred != true:
                    error = {
                        'text': text[:200],  # First 200 chars
                        'true_label': 'FAKE' if true == 1 else 'REAL',
                        'predicted_label': 'FAKE' if pred == 1 else 'REAL',
                        'confidence': confidence,
                        'error_type': 'False Positive' if true == 0 else 'False Negative'
                    }
                    
                    if true == 0:  # Real but predicted Fake
                        false_positives.append(error)
                    else:  # Fake but predicted Real
                        false_negatives.append(error)
            
            if (batch_idx + 1) % 5 == 0:
                print(f"  Processed {batch_idx + 1} batches...")
    
    return false_positives[:num_examples//2], false_negatives[:num_examples//2]

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load model
    from model import MultiModalFakeNewsDetector
    from data_loader_online import create_online_data_loaders
    
    print("\nLoading model...")
    model = MultiModalFakeNewsDetector(roberta_model_name='roberta-base', num_classes=2, hidden_dim=256, dropout=0.3)
    checkpoint = torch.load('./checkpoints_quick/best_model.pt', weights_only=False, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    print("✓ Model loaded")
    
    # Load test data
    print("\nLoading test data...")
    _, _, test_loader = create_online_data_loaders(
        train_path='./data/small/train.csv',
        val_path='./data/small/val.csv',
        test_path='./data/small/test.csv',
        cache_dir='data/image_cache_small',
        batch_size=32,
        num_workers=0
    )
    print("✓ Test data loaded")
    
    # Analyze errors
    false_positives, false_negatives = analyze_errors(model, test_loader, device, num_examples=50)
    
    # Create results directory
    os.makedirs('./results_quick', exist_ok=True)
    
    # Save to CSV
    fp_df = pd.DataFrame(false_positives)
    fn_df = pd.DataFrame(false_negatives)
    
    fp_df.to_csv('./results_quick/false_positives.csv', index=False)
    fn_df.to_csv('./results_quick/false_negatives.csv', index=False)
    
    print("ERROR ANALYSIS RESULTS")
    
    print(f"\nFalse Positives (Real classified as Fake): {len(false_positives)} examples")
    print(f"False Negatives (Fake classified as Real): {len(false_negatives)} examples")
    
    # Print sample false positives
    print("SAMPLE FALSE POSITIVES (Real news misclassified as Fake)")
    
    for i, fp in enumerate(false_positives[:3]):
        print(f"\nExample {i+1}:")
        print(f"  Text: {fp['text'][:150]}...")
        print(f"  True Label: {fp['true_label']}")
        print(f"  Predicted: {fp['predicted_label']}")
        print(f"  Confidence: {fp['confidence']:.2%}")
    
    print("SAMPLE FALSE NEGATIVES (Fake news misclassified as Real)")
    
    for i, fn in enumerate(false_negatives[:3]):
        print(f"\nExample {i+1}:")
        print(f"  Text: {fn['text'][:150]}...")
        print(f"  True Label: {fn['true_label']}")
        print(f"  Predicted: {fn['predicted_label']}")
        print(f"  Confidence: {fn['confidence']:.2%}")
    
    # Analysis
    print("\n✓ Error analysis saved to ./results_quick/")
    print("  - false_positives.csv")
    print("  - false_negatives.csv")

if __name__ == "__main__":
    main()