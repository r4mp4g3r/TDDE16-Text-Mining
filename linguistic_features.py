import numpy as np
import torch
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import textstat
import re
from typing import List, Dict
import nltk

# Download required NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


class LinguisticFeatureExtractor:

    def __init__(self):
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.emotion_words = {
            'anger': ['angry', 'furious', 'outraged', 'mad', 'rage', 'hate', 'hatred'],
            'fear': ['afraid', 'scared', 'terrified', 'fear', 'panic', 'worried', 'anxious'],
            'joy': ['happy', 'joyful', 'excited', 'thrilled', 'delighted', 'pleased'],
            'sadness': ['sad', 'depressed', 'miserable', 'unhappy', 'grief', 'sorrow']
        }
        
        # Superlatives and absolutes
        self.superlatives = ['best', 'worst', 'greatest', 'most', 'least', 'always', 
                            'never', 'everyone', 'nobody', 'everything', 'nothing']
        
        # Urgency words
        self.urgency_words = ['urgent', 'breaking', 'shocking', 'alert', 'warning', 
                             'immediately', 'now', 'must', 'emergency']
    
    def extract_features(self, text: str) -> np.ndarray:
        features = []
        
        # 1. Sentiment extremity (absolute value of compound score)
        sentiment_scores = self.sentiment_analyzer.polarity_scores(text)
        sentiment_extremity = abs(sentiment_scores['compound'])
        features.append(sentiment_extremity)
        
        # 2. Positive sentiment score
        features.append(sentiment_scores['pos'])
        
        # 3. Negative sentiment score
        features.append(sentiment_scores['neg'])
        
        # 4. Emotional word count (normalized by text length)
        emotion_count = self._count_emotional_words(text)
        text_length = len(text.split())
        emotion_ratio = emotion_count / max(text_length, 1)
        features.append(emotion_ratio)
        
        # 5. Readability (Flesch-Kincaid Grade Level)
        try:
            readability = textstat.flesch_kincaid_grade(text)
        except:
            readability = 0
        features.append(readability / 20.0)  # Normalize to [0, 1] range
        
        # 6. Superlatives and absolutes count (normalized)
        superlative_count = self._count_pattern_words(text, self.superlatives)
        superlative_ratio = superlative_count / max(text_length, 1)
        features.append(superlative_ratio)
        
        # 7. Urgency words count (normalized)
        urgency_count = self._count_pattern_words(text, self.urgency_words)
        urgency_ratio = urgency_count / max(text_length, 1)
        features.append(urgency_ratio)
        
        # 8. Exclamation marks count (normalized)
        exclamation_count = text.count('!')
        exclamation_ratio = exclamation_count / max(text_length, 1)
        features.append(exclamation_ratio)
        
        # 9. Question marks count (normalized)
        question_count = text.count('?')
        question_ratio = question_count / max(text_length, 1)
        features.append(question_ratio)
        
        # 10. Capital letters ratio (excluding first letter of sentences)
        capital_ratio = self._calculate_capital_ratio(text)
        features.append(capital_ratio)
        
        return np.array(features, dtype=np.float32)
    
    def extract_batch_features(self, texts: List[str]) -> torch.Tensor:
        features = [self.extract_features(text) for text in texts]
        return torch.tensor(np.stack(features), dtype=torch.float32)
    
    def _count_emotional_words(self, text: str) -> int:
        text_lower = text.lower()
        count = 0
        for emotion_category in self.emotion_words.values():
            for word in emotion_category:
                count += text_lower.count(word)
        return count
    
    def _count_pattern_words(self, text: str, pattern_words: List[str]) -> int:
        text_lower = text.lower()
        count = 0
        for word in pattern_words:
            # Use word boundaries to avoid partial matches
            count += len(re.findall(r'\b' + word + r'\b', text_lower))
        return count
    
    def _calculate_capital_ratio(self, text: str) -> float:
        # Remove first letter of each sentence
        sentences = text.split('. ')
        modified_text = '. '.join([s[1:] if len(s) > 1 else s for s in sentences])
        
        if len(modified_text) == 0:
            return 0.0
        
        capital_count = sum(1 for c in modified_text if c.isupper())
        return capital_count / len(modified_text)
    
    def get_feature_names(self) -> List[str]:
        return [
            'sentiment_extremity',
            'positive_sentiment',
            'negative_sentiment',
            'emotion_ratio',
            'readability',
            'superlative_ratio',
            'urgency_ratio',
            'exclamation_ratio',
            'question_ratio',
            'capital_ratio'
        ]
    
    def analyze_features(self, text: str) -> Dict[str, float]:
        features = self.extract_features(text)
        feature_names = self.get_feature_names()
        return dict(zip(feature_names, features))


def compare_real_vs_fake_features(real_texts: List[str], fake_texts: List[str]):
    extractor = LinguisticFeatureExtractor()
    
    # Extract features
    real_features = np.array([extractor.extract_features(text) for text in real_texts])
    fake_features = np.array([extractor.extract_features(text) for text in fake_texts])
    
    # Calculate means
    real_means = real_features.mean(axis=0)
    fake_means = fake_features.mean(axis=0)
    
    # Print comparison
    feature_names = extractor.get_feature_names()
    print("Feature Comparison: Real vs. Fake News")
    print("=" * 60)
    print(f"{'Feature':<25} {'Real':<15} {'Fake':<15} {'Diff':<10}")
    print("-" * 60)
    
    for i, name in enumerate(feature_names):
        diff = fake_means[i] - real_means[i]
        print(f"{name:<25} {real_means[i]:<15.4f} {fake_means[i]:<15.4f} {diff:<10.4f}")


if __name__ == "__main__":
    print("Linguistic Feature Extraction Module")
    print("=" * 50)
    
    # Test with example texts
    real_news_example = """
    The Federal Reserve announced today that it will maintain interest rates 
    at their current levels. According to Chair Powell, the decision reflects 
    ongoing economic stability and moderate inflation.
    """
    
    fake_news_example = """
    BREAKING: SHOCKING revelation! The government is hiding the TRUTH about 
    vaccines! Everyone MUST read this NOW! This is the WORST cover-up in history!
    You won't BELIEVE what they're doing!!!
    """
    
    extractor = LinguisticFeatureExtractor()
    
    print("\nReal News Features:")
    real_features = extractor.analyze_features(real_news_example)
    for feature, value in real_features.items():
        print(f"  {feature}: {value:.4f}")
    
    print("\nFake News Features:")
    fake_features = extractor.analyze_features(fake_news_example)
    for feature, value in fake_features.items():
        print(f"  {feature}: {value:.4f}")
    
    print("\nKey Differences:")
    print(f"  Sentiment Extremity: {fake_features['sentiment_extremity'] - real_features['sentiment_extremity']:.4f}")
    print(f"  Urgency Ratio: {fake_features['urgency_ratio'] - real_features['urgency_ratio']:.4f}")
    print(f"  Exclamation Ratio: {fake_features['exclamation_ratio'] - real_features['exclamation_ratio']:.4f}")
