import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
import numpy as np
from tqdm import tqdm
import os
from typing import Dict, Tuple
import json
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

from model import MultiModalFakeNewsDetector, TextOnlyBaseline, compute_clip_consistency
from linguistic_features import LinguisticFeatureExtractor
import clip


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = 'cuda',
        learning_rate: float = 2e-5,
        num_epochs: int = 5,
        warmup_steps: int = 0,
        output_dir: str = './checkpoints',
        patience: int = 3
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_epochs = num_epochs
        self.output_dir = output_dir
        self.patience = patience
        
        # Initialize feature extractor and CLIP
        self.feature_extractor = LinguisticFeatureExtractor()
        self.clip_model, _ = clip.load("ViT-B/32", device=device)
        
        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )
        
        # Scheduler
        total_steps = len(train_loader) * num_epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_f1': []
        }
        
        # Early stopping
        self.best_val_f1 = 0.0
        self.epochs_without_improvement = 0
        
        os.makedirs(output_dir, exist_ok=True)
    
    def train_epoch(self) -> Tuple[float, float]:
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        progress_bar = tqdm(self.train_loader, desc="Training")
        
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            texts = batch['text']
            
            # Extract linguistic features
            linguistic_features = self.feature_extractor.extract_batch_features(texts)
            linguistic_features = linguistic_features.to(self.device)
            
            # Compute CLIP consistency
            consistency_scores = compute_clip_consistency(
                texts, images, self.clip_model, self.device
            ).to(self.device)
            
            # Debug: Check dimensions (only once)
            if len(all_preds) == 0:
                print(f"\nDEBUG - Feature Dimensions:")
                print(f"  Consistency scores shape: {consistency_scores.shape}")
                print(f"  Linguistic features shape: {linguistic_features.shape}")
                print(f"  Combined shape: {torch.cat([consistency_scores, linguistic_features], dim=1).shape}")
            
            # Forward pass
            self.optimizer.zero_grad()
            
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                images=images,
                linguistic_features=torch.cat([consistency_scores, linguistic_features], dim=1)
            )
            
            logits = outputs['logits']
            loss = self.criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()
            
            # Track metrics
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        
        return avg_loss, accuracy
    
    def validate(self) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                # Move batch to device
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
                loss = self.criterion(logits, labels)
                
                # Track metrics
                total_loss += loss.item()
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of fake class
        
        # Calculate metrics
        avg_loss = total_loss / len(self.val_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='binary'
        )
        
        try:
            auc = roc_auc_score(all_labels, all_probs)
        except:
            auc = 0.0
        
        metrics = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc
        }
        
        return metrics
    
    def train(self):
        print(f"Starting training for {self.num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        print("=" * 60)
        
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_acc'].append(val_metrics['accuracy'])
            self.history['val_f1'].append(val_metrics['f1'])
            
            # Print metrics
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.4f}")
            print(f"Val Precision: {val_metrics['precision']:.4f} | Val Recall: {val_metrics['recall']:.4f}")
            print(f"Val F1: {val_metrics['f1']:.4f} | Val AUC: {val_metrics['auc']:.4f}")
            
            # Save checkpoint if best model
            if val_metrics['f1'] > self.best_val_f1:
                self.best_val_f1 = val_metrics['f1']
                self.epochs_without_improvement = 0
                self.save_checkpoint('best_model.pt', epoch, val_metrics)
                print(f"✓ New best model saved! (F1: {self.best_val_f1:.4f})")
            else:
                self.epochs_without_improvement += 1
                print(f"No improvement for {self.epochs_without_improvement} epoch(s)")
            
            # Early stopping
            if self.epochs_without_improvement >= self.patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break
        
        # Save final model and history
        self.save_checkpoint('final_model.pt', epoch, val_metrics)
        self.save_history()
        
        print("\n" + "=" * 60)
        print(f"Training complete! Best F1: {self.best_val_f1:.4f}")
        print(f"Models saved to: {self.output_dir}")
    
    def save_checkpoint(self, filename: str, epoch: int, metrics: Dict):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'history': self.history
        }
        path = os.path.join(self.output_dir, filename)
        torch.save(checkpoint, path)
    
    def save_history(self):
        """Save training history to JSON"""
        path = os.path.join(self.output_dir, 'training_history.json')
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)


if __name__ == "__main__":
    print("Training Script for Multi-Modal Misinformation Detection")
    
