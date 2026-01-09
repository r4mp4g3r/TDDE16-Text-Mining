import sys

print("INSTALLATION VERIFICATION")

print(f"\nPython version: {sys.version}")
print(f"Python executable: {sys.executable}")

# Test PyTorch
try:
    import torch
    print(f"\n✓ PyTorch {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  GPU device: {torch.cuda.get_device_name(0)}")
    
    # Check for Mac MPS
    if hasattr(torch.backends, 'mps'):
        print(f"  MPS available: {torch.backends.mps.is_available()}")
        if torch.backends.mps.is_available():
            print(f"  → Can use Apple Silicon GPU acceleration!")
except ImportError as e:
    print(f"\n✗ PyTorch installation failed: {e}")
    sys.exit(1)

# Test Transformers
try:
    from transformers import RobertaTokenizer, RobertaModel
    print("\n✓ Transformers installed")
    
    # Try loading a model (this will download it if not cached)
    print("  Testing RoBERTa model loading...")
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    print("  → RoBERTa tokenizer loaded successfully")
except ImportError as e:
    print(f"\n✗ Transformers installation failed: {e}")
    sys.exit(1)
except Exception as e:
    print(f"  ⚠ Warning: Could not load RoBERTa model: {e}")

# Test CLIP
try:
    import clip
    print("\n✓ CLIP installed")
    
    # Try loading CLIP model
    print("  Testing CLIP model loading...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    print(f"  → CLIP model loaded successfully on {device}")
except ImportError as e:
    print(f"\n✗ CLIP installation failed: {e}")
    sys.exit(1)
except Exception as e:
    print(f"  ⚠ Warning: Could not load CLIP model: {e}")

# Test Data Processing
try:
    import pandas as pd
    import numpy as np
    from PIL import Image
    print("\n✓ Data processing libraries installed")
    print(f"  Pandas: {pd.__version__}")
    print(f"  NumPy: {np.__version__}")
except ImportError as e:
    print(f"\n✗ Data processing libraries failed: {e}")
    sys.exit(1)

# Test NLP Utilities
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    import textstat
    import nltk
    print("\n✓ NLP utilities installed")
    
    # Test VADER
    analyzer = SentimentIntensityAnalyzer()
    test_score = analyzer.polarity_scores("This is a test")
    print("  → VADER sentiment analyzer working")
    
    # Check NLTK data
    try:
        nltk.data.find('tokenizers/punkt')
        print("  → NLTK punkt data available")
    except LookupError:
        print("  ⚠ Warning: NLTK punkt data not found")
        print("  Run: python -c \"import nltk; nltk.download('punkt')\"")
except ImportError as e:
    print(f"\n✗ NLP utilities installation failed: {e}")
    sys.exit(1)

# Test Scikit-learn
try:
    from sklearn.metrics import accuracy_score
    import sklearn
    print(f"\n✓ Scikit-learn {sklearn.__version__}")
except ImportError as e:
    print(f"\n✗ Scikit-learn installation failed: {e}")
    sys.exit(1)

# Test Visualization
try:
    import matplotlib
    import matplotlib.pyplot as plt
    import seaborn as sns
    print(f"\n✓ Visualization libraries installed")
    print(f"  Matplotlib: {matplotlib.__version__}")
    print(f"  Seaborn: {sns.__version__}")
except ImportError as e:
    print(f"\n✗ Visualization libraries failed: {e}")
    sys.exit(1)

# Test project modules
print("TESTING PROJECT MODULES")

try:
    from model import MultiModalFakeNewsDetector, TextOnlyBaseline
    print("\n✓ Model module imported successfully")
except ImportError as e:
    print(f"\n✗ Model module import failed: {e}")

try:
    from linguistic_features import LinguisticFeatureExtractor
    print("✓ Linguistic features module imported successfully")
    
    # Test feature extraction
    extractor = LinguisticFeatureExtractor()
    test_text = "This is a test sentence for feature extraction."
    features = extractor.extract_features(test_text)
    print(f"  → Extracted {len(features)} features")
except ImportError as e:
    print(f"\n✗ Linguistic features module import failed: {e}")
except Exception as e:
    print(f"  ⚠ Warning: Feature extraction test failed: {e}")

try:
    from data_loader import MultiModalDataset
    print("✓ Data loader module imported successfully")
except ImportError as e:
    print(f"\n✗ Data loader module import failed: {e}")

try:
    from train import Trainer
    print("✓ Training module imported successfully")
except ImportError as e:
    print(f"\n✗ Training module import failed: {e}")

try:
    from evaluate import ModelEvaluator
    print("✓ Evaluation module imported successfully")
except ImportError as e:
    print(f"\n✗ Evaluation module import failed: {e}")

# Summary
print("SUMMARY")

# Determine device
if torch.cuda.is_available():
    device_info = "CUDA GPU (NVIDIA)"
    device_name = torch.cuda.get_device_name(0)
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device_info = "MPS GPU (Apple Silicon)"
    device_name = "Apple M-series chip"
else:
    device_info = "CPU only"
    device_name = "No GPU acceleration"

print(f"\n✅ All core dependencies installed successfully!")
print(f"\nDevice: {device_info}")
print(f"  {device_name}")

