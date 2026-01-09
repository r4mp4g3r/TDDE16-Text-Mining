import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, roc_curve, auc, classification_report
)
from typing import List, Dict, Tuple
import pandas as pd
from tqdm import tqdm
import os

from model import MultiModalFakeNewsDetector, compute_clip_consistency
from linguistic_features import LinguisticFeatureExtractor
import clip


class ModelEvaluator:
    def __init__(
        self,
        model: MultiModalFakeNewsDetector,
        test_loader,
        device: str = 'cuda',
        output_dir: str = './results'
    ):
        self.model = model.to(device)
        self.test_loader = test_loader
        self.device = device
        self.output_dir = output_dir
        
        self.feature_extractor = LinguisticFeatureExtractor()
        self.clip_model, _ = clip.load("ViT-B/32", device=device)
        
        os.makedirs(output_dir, exist_ok=True)
    
    def evaluate(self) -> Dict[str, float]:
        print("Running evaluation...")
        self.model.eval()
        
        all_preds = []
        all_labels = []
        all_probs = []
        all_texts = []
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                texts = batch['text']
                
                # Extract features
                linguistic_features = self.feature_extractor.extract_batch_features(texts)
                linguistic_features = linguistic_features.to(self.device)
                
                consistency_scores = compute_clip_consistency(
                    texts, images, self.clip_model, self.device
                ).to(self.device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    images=images,
                    linguistic_features=torch.cat([consistency_scores, linguistic_features], dim=1)
                )
                
                logits = outputs['logits']
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())
                all_texts.extend(texts)
        
        # Calculate metrics
        metrics = self._calculate_metrics(all_labels, all_preds, all_probs)
        
        # Save predictions for error analysis
        self.predictions_df = pd.DataFrame({
            'text': all_texts,
            'true_label': all_labels,
            'predicted_label': all_preds,
            'probability': all_probs
        })
        self.predictions_df.to_csv(
            os.path.join(self.output_dir, 'predictions.csv'),
            index=False
        )
        
        # Print metrics
        self._print_metrics(metrics)
        
        # Generate visualizations
        self._plot_confusion_matrix(all_labels, all_preds)
        self._plot_roc_curve(all_labels, all_probs)
        
        return metrics
    
    def _calculate_metrics(
        self,
        labels: List[int],
        preds: List[int],
        probs: List[float]
    ) -> Dict[str, float]:
        accuracy = accuracy_score(labels, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average='binary'
        )
        
        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, _ = \
            precision_recall_fscore_support(labels, preds, average=None)
        
        # ROC AUC
        fpr, tpr, _ = roc_curve(labels, probs)
        roc_auc = auc(fpr, tpr)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': roc_auc,
            'precision_real': precision_per_class[0],
            'recall_real': recall_per_class[0],
            'f1_real': f1_per_class[0],
            'precision_fake': precision_per_class[1],
            'recall_fake': recall_per_class[1],
            'f1_fake': f1_per_class[1]
        }
        
        return metrics
    
    def _print_metrics(self, metrics: Dict[str, float]):
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)
        print(f"Overall Accuracy: {metrics['accuracy']:.4f}")
        print(f"Overall Precision: {metrics['precision']:.4f}")
        print(f"Overall Recall: {metrics['recall']:.4f}")
        print(f"Overall F1-Score: {metrics['f1']:.4f}")
        print(f"ROC AUC: {metrics['auc']:.4f}")
        print("\nPer-Class Metrics:")
        print(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
        print("-" * 60)
        print(f"{'Real (0)':<15} {metrics['precision_real']:<12.4f} "
              f"{metrics['recall_real']:<12.4f} {metrics['f1_real']:<12.4f}")
        print(f"{'Fake (1)':<15} {metrics['precision_fake']:<12.4f} "
              f"{metrics['recall_fake']:<12.4f} {metrics['f1_fake']:<12.4f}")
        print("=" * 60)
    
    def _plot_confusion_matrix(self, labels: List[int], preds: List[int]):
        cm = confusion_matrix(labels, preds)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Real', 'Fake'],
            yticklabels=['Real', 'Fake']
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'confusion_matrix.png'), dpi=300)
        plt.close()
        
        print(f"✓ Confusion matrix saved to {self.output_dir}/confusion_matrix.png")
    
    def _plot_roc_curve(self, labels: List[int], probs: List[float]):
        fpr, tpr, _ = roc_curve(labels, probs)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'roc_curve.png'), dpi=300)
        plt.close()
        
        print(f"✓ ROC curve saved to {self.output_dir}/roc_curve.png")
    
    def error_analysis(self, num_examples: int = 50):
        print("\nPerforming error analysis...")
        
        # Get misclassified examples
        errors = self.predictions_df[
            self.predictions_df['true_label'] != self.predictions_df['predicted_label']
        ]
        
        # Separate false positives and false negatives
        false_positives = errors[errors['predicted_label'] == 1]  # Real predicted as Fake
        false_negatives = errors[errors['predicted_label'] == 0]  # Fake predicted as Real
        
        print(f"\nTotal misclassifications: {len(errors)}")
        print(f"False Positives (Real → Fake): {len(false_positives)}")
        print(f"False Negatives (Fake → Real): {len(false_negatives)}")
        
        # Save error examples
        error_report = []
        
        # Analyze false positives
        print("\n" + "=" * 60)
        print("FALSE POSITIVES (Real news classified as Fake):")
        print("=" * 60)
        for idx, row in false_positives.head(min(num_examples // 2, len(false_positives))).iterrows():
            print(f"\nExample {idx}:")
            print(f"Text: {row['text'][:200]}...")
            print(f"Confidence: {row['probability']:.4f}")
            error_report.append({
                'type': 'False Positive',
                'text': row['text'],
                'confidence': row['probability']
            })
        
        # Analyze false negatives
        print("\n" + "=" * 60)
        print("FALSE NEGATIVES (Fake news classified as Real):")
        print("=" * 60)
        for idx, row in false_negatives.head(min(num_examples // 2, len(false_negatives))).iterrows():
            print(f"\nExample {idx}:")
            print(f"Text: {row['text'][:200]}...")
            print(f"Confidence: {1 - row['probability']:.4f}")
            error_report.append({
                'type': 'False Negative',
                'text': row['text'],
                'confidence': 1 - row['probability']
            })
        
        # Save error report
        pd.DataFrame(error_report).to_csv(
            os.path.join(self.output_dir, 'error_analysis.csv'),
            index=False
        )
        print(f"\n✓ Error analysis saved to {self.output_dir}/error_analysis.csv")
    
    def visualize_attention(self, text: str, tokenizer, max_words: int = 50):
        self.model.eval()
        
        # Tokenize
        encoding = tokenizer(
            text,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Extract attention
        with torch.no_grad():
            attention_weights = self.model.extract_attention(input_ids, attention_mask)
        
        # Get tokens
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
        attention_weights = attention_weights[0].cpu().numpy()
        
        # Filter out padding and special tokens
        valid_indices = [i for i, token in enumerate(tokens) 
                        if token not in ['<pad>', '<s>', '</s>']]
        tokens = [tokens[i] for i in valid_indices][:max_words]
        attention_weights = attention_weights[valid_indices][:max_words]
        
        # Normalize attention weights
        attention_weights = attention_weights / attention_weights.max()
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(12, 6))
        colors = plt.cm.Reds(attention_weights)
        
        y_pos = np.arange(len(tokens))
        ax.barh(y_pos, attention_weights, color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(tokens)
        ax.invert_yaxis()
        ax.set_xlabel('Attention Weight')
        ax.set_title('Attention Visualization')
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, 'attention_visualization.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Attention visualization saved to {output_path}")
        
        return tokens, attention_weights


if __name__ == "__main__":
    
    print("\nTo use this module:")
    print("1. Load your trained model")
    print("2. Create a ModelEvaluator instance")
    print("3. Call evaluator.evaluate() for full evaluation")
    print("4. Call evaluator.error_analysis() for error analysis")
    print("5. Call evaluator.visualize_attention() for attention visualization")
