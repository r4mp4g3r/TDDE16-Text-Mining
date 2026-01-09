import torch
import numpy as np
import json
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import clip

# Fix PyTorch 2.6
try:
    torch.serialization.add_safe_globals([np._core.multiarray.scalar])
except:
    torch.serialization.add_safe_globals([np.core.multiarray.scalar])

def evaluate_model(model, test_loader, device='cuda', use_clip=True, use_linguistic=True, feature_extractor=None, clip_model=None):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            texts = batch['text']
            batch_size = input_ids.shape[0]
            
            # Build feature vector based on ablation variant
            features_list = []
            
            # CLIP consistency (1 dim)
            if use_clip and clip_model is not None:
                from model import compute_clip_consistency
                consistency_scores = compute_clip_consistency(texts, images, clip_model, device).to(device)
                features_list.append(consistency_scores)
            else:
                # Add zero features if not using CLIP
                features_list.append(torch.zeros(batch_size, 1).to(device))
            
            # Linguistic features (10 dim)
            if use_linguistic and feature_extractor is not None:
                linguistic_features = feature_extractor.extract_batch_features(texts).to(device)
                features_list.append(linguistic_features)
            else:
                # Add zero features if not using linguistic
                features_list.append(torch.zeros(batch_size, 10).to(device))
            
            # Concatenate all features (should be 11 dim total: 1 CLIP + 10 linguistic)
            combined_features = torch.cat(features_list, dim=1)
            
            output = model(input_ids, attention_mask, images, combined_features)
            
            if isinstance(output, dict):
                logits = output.get('logits', list(output.values())[0])
            else:
                logits = output
            
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    metrics = {
        'accuracy': float(accuracy_score(all_labels, all_preds)),
        'precision': float(precision_score(all_labels, all_preds, zero_division=0)),
        'recall': float(recall_score(all_labels, all_preds, zero_division=0)),
        'f1': float(f1_score(all_labels, all_preds, zero_division=0)),
        'auc': float(roc_auc_score(all_labels, all_probs))
    }
    
    return metrics

def evaluate_text_only_model(model, test_loader, device='cuda'):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            output = model(input_ids, attention_mask)
            
            if isinstance(output, dict):
                logits = output.get('logits', list(output.values())[0])
            else:
                logits = output
            
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    metrics = {
        'accuracy': float(accuracy_score(all_labels, all_preds)),
        'precision': float(precision_score(all_labels, all_preds, zero_division=0)),
        'recall': float(recall_score(all_labels, all_preds, zero_division=0)),
        'f1': float(f1_score(all_labels, all_preds, zero_division=0)),
        'auc': float(roc_auc_score(all_labels, all_probs))
    }
    
    return metrics

def main():
    
    print("ABLATION STUDY")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Fix PyTorch 2.6
    try:
        torch.serialization.add_safe_globals([np._core.multiarray.scalar])
    except:
        torch.serialization.add_safe_globals([np.core.multiarray.scalar])
    
    # Load test data
    from data_loader_online import create_online_data_loaders
    from model import MultiModalFakeNewsDetector, TextOnlyBaseline
    from linguistic_features import LinguisticFeatureExtractor
    
    print("\nLoading test data...")
    _, _, test_loader = create_online_data_loaders(
        train_path='./data/processed/train.csv',
        val_path='./data/processed/val.csv',
        test_path='./data/processed/test.csv',
        cache_dir='data/image_cache',
        batch_size=32,
        num_workers=0
    )
    
    # Initialize feature extractors
    print("Initializing feature extractors...")
    feature_extractor = LinguisticFeatureExtractor()
    clip_model, _ = clip.load("ViT-B/32", device=device)
    
    results = {}
    
    # 1. Full Model (Text + CLIP + Linguistic)
    print("1. Evaluating FULL MODEL (Text + CLIP + Linguistic)...")
    
    model = MultiModalFakeNewsDetector(roberta_model_name='roberta-base', num_classes=2, hidden_dim=256, dropout=0.3)
    checkpoint_path = './checkpoints_quick/best_model.pt'
    if not os.path.exists(checkpoint_path):
        checkpoint_path = './results/best_model.pt'
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        
        metrics = evaluate_model(
            model, test_loader, device, 
            use_clip=True, use_linguistic=True,
            feature_extractor=feature_extractor, clip_model=clip_model
        )
        results['Full Model'] = metrics
        print(f"  ✓ Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}, AUC: {metrics['auc']:.4f}")
    else:
        print(f"  ✗ Checkpoint not found at {checkpoint_path}")
        print("  Skipping full model evaluation")
    
    # 2. Text-Only Model
    print("2. Evaluating TEXT-ONLY MODEL (no CLIP, no linguistic)...")
    
    # Try to load text-only model if it exists, otherwise create one
    text_only_checkpoint_path = './checkpoints_quick/text_only_model.pt'
    if not os.path.exists(text_only_checkpoint_path):
        text_only_checkpoint_path = './results/text_only_model.pt'
    
    if os.path.exists(text_only_checkpoint_path):
        text_only_model = TextOnlyBaseline(roberta_model_name='roberta-base', num_classes=2, hidden_dim=256, dropout=0.3)
        checkpoint = torch.load(text_only_checkpoint_path, weights_only=False, map_location=device)
        text_only_model.load_state_dict(checkpoint['model_state_dict'])
        text_only_model.to(device)
        
        metrics = evaluate_text_only_model(text_only_model, test_loader, device)
        results['Text-Only'] = metrics
        print(f"  ✓ Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}, AUC: {metrics['auc']:.4f}")
    else:
        print("  Note: Text-only model checkpoint not found.")
        print("  Skipping text-only evaluation.")
        results['Text-Only'] = {
            'accuracy': None,
            'precision': None,
            'recall': None,
            'f1': None,
            'auc': None,
            'note': 'Requires separately trained text-only model'
        }
    
    # 3. Text + Linguistic (no CLIP)
    print("3. Evaluating TEXT + LINGUISTIC MODEL (no CLIP)...")
    
    if os.path.exists(checkpoint_path):
        model = MultiModalFakeNewsDetector(roberta_model_name='roberta-base', num_classes=2, hidden_dim=256, dropout=0.3)
        checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        
        metrics = evaluate_model(
            model, test_loader, device,
            use_clip=False, use_linguistic=True,
            feature_extractor=feature_extractor, clip_model=None
        )
        results['Text + Linguistic'] = metrics
        print(f"  ✓ Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}, AUC: {metrics['auc']:.4f}")
    else:
        print("  ✗ Cannot evaluate: Full model checkpoint not found")
        results['Text + Linguistic']
    
    # 4. Text + CLIP (no linguistic)
    print("4. Evaluating TEXT + CLIP MODEL (no linguistic)...")
    
    if os.path.exists(checkpoint_path):
        model = MultiModalFakeNewsDetector(roberta_model_name='roberta-base', num_classes=2, hidden_dim=256, dropout=0.3)
        checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        
        metrics = evaluate_model(
            model, test_loader, device,
            use_clip=True, use_linguistic=False,
            feature_extractor=None, clip_model=clip_model
        )
        results['Text + CLIP'] = metrics
        print(f"  ✓ Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}, AUC: {metrics['auc']:.4f}")
    else:
        print("  ✗ Cannot evaluate: Full model checkpoint not found")
        results['Text + CLIP']
    
    # Print results table
    print("ABLATION STUDY RESULTS")
    print(f"\n{'Model':<25} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'AUC':<12}")
    print("-" * 85)
    
    for model_name, metrics in results.items():
        if 'note' in metrics:
            print(f"{model_name:<25} {'N/A (see note)':<12}")
        elif metrics.get('accuracy') is not None:
            print(f"{model_name:<25} {metrics['accuracy']:<12.4f} {metrics['precision']:<12.4f} {metrics['recall']:<12.4f} {metrics['f1']:<12.4f} {metrics['auc']:<12.4f}")
        else:
            print(f"{model_name:<25} {'N/A':<12}")
    
    # Save results
    os.makedirs('./results_quick', exist_ok=True)
    with open('./results_quick/ablation_study.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create visualization (only for models with valid results)
    valid_results = {k: v for k, v in results.items() 
                     if 'note' not in v and v.get('accuracy') is not None}
    
    if len(valid_results) > 0:
        models = list(valid_results.keys())
        f1_scores = [valid_results[m]['f1'] for m in models]
        accuracies = [valid_results[m]['accuracy'] for m in models]
        
        # Determine color scheme
        colors = []
        for m in models:
            if 'Full' in m:
                colors.append('green')
            elif 'Text-Only' in m:
                colors.append('red')
            elif 'Linguistic' in m:
                colors.append('orange')
            elif 'CLIP' in m:
                colors.append('blue')
            else:
                colors.append('gray')
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # F1 scores
        bars1 = ax1.bar(range(len(models)), f1_scores, color=colors)
        ax1.set_xticks(range(len(models)))
        ax1.set_xticklabels(models, rotation=45, ha='right')
        ax1.set_ylabel('F1-Score')
        ax1.set_title('Ablation Study: F1-Score Comparison')
        ax1.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars1, f1_scores)):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Accuracies
        bars2 = ax2.bar(range(len(models)), accuracies, color=colors)
        ax2.set_xticks(range(len(models)))
        ax2.set_xticklabels(models, rotation=45, ha='right')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Ablation Study: Accuracy Comparison')
        ax2.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars2, accuracies)):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig('./results_quick/ablation_study.png', dpi=100, bbox_inches='tight')
        plt.close()
        
        print("\n✓ Ablation study results saved to ./results_quick/ablation_study.json")
        print("✓ Visualization saved to ./results_quick/ablation_study.png")
    else:
        print("\n⚠ No valid results to visualize. Check that model checkpoints exist.")
    

if __name__ == "__main__":
    main()