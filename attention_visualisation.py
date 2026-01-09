import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import RobertaTokenizer
import os

# Fix PyTorch 2.6
try:
    torch.serialization.add_safe_globals([np._core.multiarray.scalar])
except:
    torch.serialization.add_safe_globals([np.core.multiarray.scalar])

def visualize_attention(text, attention_weights, tokenizer, output_path=None):
    
    # Tokenize text
    tokens = tokenizer.tokenize(text)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create heatmap
    im = ax.imshow(attention_weights, cmap='YlOrRd', aspect='auto')
    
    # Set labels
    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=45, ha='right')
    ax.set_ylabel('Attention Head')
    ax.set_title('Attention Weights Visualization')
    
    # Add colorbar
    plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        print(f"✓ Saved to {output_path}")
    
    plt.close()

def extract_attention_from_model(model, input_ids, attention_mask, device='cuda'):
    model.eval()
    
    with torch.no_grad():
        # Forward pass with output_attentions=True
        outputs = model.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True
        )
        
        # Get attention from last layer
        attention = outputs.attentions[-1]  # Shape: [batch, heads, seq_len, seq_len]
        
        # Average over heads and batch
        attention = attention.mean(dim=1)[0]  # [seq_len, seq_len]
        
        # Get attention to [CLS] token (first token)
        cls_attention = attention[0]  # Attention from [CLS] to all tokens
        
        return cls_attention.cpu().numpy()

def main():
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load model and tokenizer
    from model import MultiModalFakeNewsDetector
    from transformers import RobertaTokenizer
    
    print("\nLoading model and tokenizer...")
    model = MultiModalFakeNewsDetector(roberta_model_name='roberta-base', num_classes=2, hidden_dim=256, dropout=0.3)
    checkpoint = torch.load('./checkpoints_quick/best_model.pt', weights_only=False, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    print("✓ Model and tokenizer loaded")
    
    # Create output directory
    os.makedirs('./results_quick/attention_visualizations', exist_ok=True)
    
    # Visualize fake news examples
    print("\nVisualizing FAKE NEWS examples...")
    for i, text in enumerate(fake_news_samples):
        print(f"  Processing fake news sample {i+1}...")
        
        # Tokenize
        encoded = tokenizer.encode_plus(
            text,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)
        
        # Extract attention
        attention = extract_attention_from_model(model, input_ids, attention_mask, device)
        
        # Visualize
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
        
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.barh(range(len(tokens[:50])), attention[:50])
        ax.set_yticks(range(len(tokens[:50])))
        ax.set_yticklabels(tokens[:50])
        ax.set_xlabel('Attention Weight')
        ax.set_title(f'Attention Weights - Fake News Example {i+1}')
        plt.tight_layout()
        plt.savefig(f'./results_quick/attention_visualizations/fake_news_{i+1}.png', dpi=100, bbox_inches='tight')
        plt.close()
        
        print(f" ✓ Saved to attention_visualizations/fake_news_{i+1}.png")
    
    # Visualize real news examples
    print("\nVisualizing REAL NEWS examples...")
    for i, text in enumerate(real_news_samples):
        print(f"  Processing real news sample {i+1}...")
        
        # Tokenize
        encoded = tokenizer.encode_plus(
            text,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)
        
        # Extract attention
        attention = extract_attention_from_model(model, input_ids, attention_mask, device)
        
        # Visualize
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
        
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.barh(range(len(tokens[:50])), attention[:50])
        ax.set_yticks(range(len(tokens[:50])))
        ax.set_yticklabels(tokens[:50])
        ax.set_xlabel('Attention Weight')
        ax.set_title(f'Attention Weights - Real News Example {i+1}')
        plt.tight_layout()
        plt.savefig(f'./results_quick/attention_visualizations/real_news_{i+1}.png', dpi=100, bbox_inches='tight')
        plt.close()
        
        print(f"    ✓ Saved to attention_visualizations/real_news_{i+1}.png")
    
    print("\n✓ All attention visualizations saved to ./results_quick/attention_visualizations/")

if __name__ == "__main__":
    main()