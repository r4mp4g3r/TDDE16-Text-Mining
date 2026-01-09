import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaConfig
import clip
from typing import Dict, Tuple


class MultiModalFakeNewsDetector(nn.Module):
    
    def __init__(
        self,
        roberta_model_name: str = 'roberta-base',
        clip_model_name: str = 'ViT-B/32',
        num_classes: int = 2,
        hidden_dim: int = 256,
        dropout: float = 0.3,
        freeze_roberta: bool = False
    ):
        super(MultiModalFakeNewsDetector, self).__init__()
        
        # Text encoder: RoBERTa
        self.roberta = RobertaModel.from_pretrained(roberta_model_name)
        if freeze_roberta:
            for param in self.roberta.parameters():
                param.requires_grad = False
        
        # Image-text consistency: CLIP
        self.clip_model, _ = clip.load(clip_model_name, device='cpu')
        for param in self.clip_model.parameters():
            param.requires_grad = False  # Keep CLIP frozen
        
        # Feature dimensions
        self.text_dim = 768  # RoBERTa hidden size
        self.consistency_dim = 1  # CLIP similarity score
        self.linguistic_dim = 10  # Custom linguistic features
        
        # Total input dimension for classifier
        self.total_dim = self.text_dim + self.consistency_dim + self.linguistic_dim
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(self.total_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        images: torch.Tensor,
        linguistic_features: torch.Tensor,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        batch_size = input_ids.size(0)
        
        # 1. Text encoding with RoBERTa
        roberta_output = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=return_attention
        )
        
        # Extract [CLS] token embedding
        text_features = roberta_output.last_hidden_state[:, 0, :]  # [batch_size, 768]
        
        # 2. Concatenate all features
        fused_features = torch.cat([
            text_features,
            linguistic_features  # This is [batch_size, 11] = 1 (consistency) + 10 (linguistic)
        ], dim=1)  # [batch_size, 768 + 11 = 779]
        
        # 4. Classification
        logits = self.classifier(fused_features)  # [batch_size, num_classes]
        
        output = {'logits': logits}
        
        if return_attention:
            output['attention_weights'] = roberta_output.attentions
        
        return output
    
    def extract_attention(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        with torch.no_grad():
            outputs = self.roberta(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True
            )
            
            # Get attention from last layer
            last_layer_attention = outputs.attentions[-1]  # [batch, heads, seq, seq]
            
            # Average across heads and take attention to [CLS] token
            avg_attention = last_layer_attention.mean(dim=1)  # [batch, seq, seq]
            cls_attention = avg_attention[:, 0, :]  # [batch, seq]
            
            return cls_attention


class TextOnlyBaseline(nn.Module):
    def __init__(
        self,
        roberta_model_name: str = 'roberta-base',
        num_classes: int = 2,
        hidden_dim: int = 256,
        dropout: float = 0.3
    ):
        super(TextOnlyBaseline, self).__init__()
        
        self.roberta = RobertaModel.from_pretrained(roberta_model_name)
        
        self.classifier = nn.Sequential(
            nn.Linear(768, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        text_features = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(text_features)
        
        return logits


def compute_clip_consistency(
    texts: list,
    images: torch.Tensor,
    clip_model,
    device: str = 'cuda'
) -> torch.Tensor:
    with torch.no_grad():
        # Tokenize texts for CLIP
        text_tokens = clip.tokenize(texts, truncate=True).to(device)
        
        # Encode
        text_features = clip_model.encode_text(text_tokens)
        image_features = clip_model.encode_image(images.to(device))
        
        # Normalize
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # Compute cosine similarity
        similarity = (text_features * image_features).sum(dim=-1, keepdim=True)
        
    return similarity.cpu()


if __name__ == "__main__":
    print("Multi-Modal Fake News Detection Model")
    
    # Test model initialization
    model = MultiModalFakeNewsDetector()
    print(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test forward pass with dummy data
    batch_size = 4
    seq_len = 128
    
    dummy_input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    dummy_attention_mask = torch.ones(batch_size, seq_len)
    dummy_images = torch.randn(batch_size, 3, 224, 224)
    dummy_linguistic = torch.randn(batch_size, 10)
    
    output = model(
        dummy_input_ids,
        dummy_attention_mask,
        dummy_images,
        dummy_linguistic
    )
    
    print(f"Output logits shape: {output['logits'].shape}")
    print("Model test passed!")
