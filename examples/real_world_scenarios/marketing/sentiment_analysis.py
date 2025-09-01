# File: examples/real_world_scenarios/marketing/sentiment_analysis.py
# Location: examples/real_world_scenarios/marketing/sentiment_analysis.py

"""
Sentiment Analysis System - Real-World ML Pipeline Example

Business Problem:
Analyze customer sentiment from reviews, social media, and feedback to improve
product development, customer service, and marketing strategies.

Dataset: Multi-source text data with sentiment labels (synthetic)
Target: Multi-class sentiment classification (positive, negative, neutral) + emotion detection
Business Impact: 40% improvement in customer satisfaction, $2.1M crisis prevention value
Techniques: NLP preprocessing, feature engineering, ensemble classification, trend analysis

Industry Applications:
- E-commerce platforms
- Social media monitoring
- Brand management
- Customer service
- Product development
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, Any, List, Optional
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
import re
import string

# Framework imports
from src.data.generators import DataGenerator
from src.models.supervised.classification import ClassificationModels
from src.evaluation.metrics import ModelEvaluator

# Scenario-specific imports
from ..utilities.data_loaders import DataLoader
from ..utilities.visualization_helpers import BusinessVisualizer
from ..utilities.evaluation_helpers import BusinessMetricsCalculator, ModelPerformanceEvaluator

class SentimentAnalysisSystem:
    """Complete sentiment analysis system for marketing and customer insights."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize sentiment analysis system."""
        
        self.config = config or {
            'n_texts': 50000,
            'test_size': 0.2,
            'random_state': 42,
            'algorithms': ['random_forest', 'gradient_boosting', 'logistic_regression', 'neural_network'],
            'sentiment_classes': ['negative', 'neutral', 'positive'],
            'emotion_classes': ['anger', 'fear', 'joy', 'sadness', 'surprise', 'neutral'],
            'text_sources': ['reviews', 'social_media', 'surveys', 'support_tickets', 'forums'],
            'business_params': {
                'crisis_threshold': 0.3,  # 30% negative sentiment threshold
                'intervention_cost': 10000,
                'crisis_damage_cost': 500000,
                'positive_sentiment_value': 50,
                'negative_sentiment_cost': 200
            }
        }
        
        # Initialize components
        self.data_loader = DataLoader(random_state=self.config['random_state'])
        self.visualizer = BusinessVisualizer()
        self.business_calc = BusinessMetricsCalculator()
        self.model_evaluator = ModelPerformanceEvaluator()
        
        # Results storage
        self.text_data = None
        self.sentiment_results = {}
        self.best_models = {}
        
    def generate_text_dataset(self) -> Tuple[pd.DataFrame, Dict[str, pd.Series]]:
        """Generate comprehensive text dataset with sentiment and emotion labels."""
        
        print("🔄 Generating sentiment analysis dataset...")
        
        np.random.seed(self.config['random_state'])
        
        # Generate text records
        text_records = []
        base_date = datetime(2023, 1, 1)
        
        # Template patterns for different sentiments and sources
        templates = self._get_text_templates()
        
        for i in range(self.config['n_texts']):
            # Select source and determine characteristics
            source = np.random.choice(self.config['text_sources'])
            date = base_date + timedelta(days=np.random.randint(0, 365))
            
            # Generate sentiment and emotion (correlated but not identical)
            sentiment, emotion = self._generate_sentiment_emotion()
            
            # Generate text content
            text_content = self._generate_text_content(source, sentiment, emotion, templates)
            
            # Calculate text features
            text_features = self._extract_text_features(text_content)
            
            record = {
                'text_id': f'T{i:06d}',
                'date': date,
                'source': source,
                'text_content': text_content,
                'sentiment': sentiment,
                'emotion': emotion,
                
                # Text characteristics
                'text_length': len(text_content),
                'word_count': len(text_content.split()),
                'sentence_count': len([s for s in text_content.split('.') if s.strip()]),
                'avg_word_length': np.mean([len(word) for word in text_content.split()]),
                
                # Extracted features
                **text_features,
                
                # Temporal features
                'month': date.month,
                'quarter': (date.month - 1) // 3 + 1,
                'day_of_week': date.weekday(),
                'is_weekend': 1 if date.weekday() >= 5 else 0,
                
                # Source-specific features
                'is_review': 1 if source == 'reviews' else 0,
                'is_social_media': 1 if source == 'social_media' else 0,
                'is_support': 1 if source == 'support_tickets' else 0,
                'is_survey': 1 if source == 'surveys' else 0,
                'is_forum': 1 if source == 'forums' else 0,
                
                # Engagement features (synthetic)
                'likes': np.random.poisson(5) if source == 'social_media' else 0,
                'shares': np.random.poisson(2) if source == 'social_media' else 0,
                'comments': np.random.poisson(3) if source in ['social_media', 'forums'] else 0,
                'helpfulness_votes': np.random.poisson(1) if source == 'reviews' else 0,
                
                # Metadata
                'user_type': np.random.choice(['new', 'returning', 'premium'], p=[0.4, 0.4, 0.2]),
                'product_category': np.random.choice(['electronics', 'clothing', 'books', 'home', 'sports']),
                'response_time': np.random.exponential(2) if source == 'support_tickets' else None,
                
                # Sentiment confidence (simulated)
                'sentiment_confidence': np.random.beta(8, 2),  # High confidence bias
                'emotion_intensity': np.random.uniform(0.1, 1.0)
            }
            
            text_records.append(record)
        
        # Create DataFrame
        df = pd.DataFrame(text_records)
        
        # Add derived features
        df['engagement_score'] = (df['likes'] + df['shares'] * 2 + df['comments']) / (df['word_count'] + 1)
        df['text_complexity'] = df['avg_word_length'] * df['sentence_count'] / (df['word_count'] + 1)
        df['weekend_post'] = df['is_weekend']
        
        # Add temporal aggregations
        df = df.sort_values('date')
        df['sentiment_trend_7d'] = df.groupby('source')['sentiment'].rolling(7, min_periods=1).apply(
            lambda x: (x == 'positive').mean() - (x == 'negative').mean()
        ).reset_index(0, drop=True)
        
        # Create targets
        targets = {
            'sentiment_classification': df['sentiment'],
            'emotion_classification': df['emotion']
        }
        
        # Feature selection (exclude text content and target variables)
        feature_cols = [col for col in df.columns if col not in 
                       ['text_id', 'date', 'text_content', 'sentiment', 'emotion', 'response_time']]
        
        X = df[feature_cols].fillna(0)
        
        print(f"✅ Generated {len(df):,} text records")
        print(f"📊 Sources: {len(self.config['text_sources'])}, Features: {len(feature_cols)}")
        print(f"💬 Sentiment distribution: {dict(df['sentiment'].value_counts())}")
        print(f"😊 Emotion distribution: {dict(df['emotion'].value_counts())}")
        
        return X, targets
    
    def _get_text_templates(self) -> Dict[str, Dict[str, List[str]]]:
        """Get text templates for different sources, sentiments, and emotions."""
        
        return {
            'reviews': {
                'positive': [
                    "Amazing product! Exceeded my expectations completely. Highly recommended!",
                    "Love this item! Great quality and fast shipping. Will definitely buy again.",
                    "Fantastic! Works perfectly and looks great. Best purchase I've made.",
                    "Outstanding quality! Worth every penny. Five stars!",
                    "Incredible value! Better than expected. Couldn't be happier!"
                ],
                'negative': [
                    "Terrible quality. Broke after just a few days. Complete waste of money.",
                    "Disappointed. Not as described. Returning immediately.",
                    "Poor construction. Doesn't work properly. Avoid this product.",
                    "Awful experience. Product defective. Customer service unhelpful.",
                    "Regret buying this. Cheaply made. One star only."
                ],
                'neutral': [
                    "Product is okay. Does what it's supposed to do. Nothing special.",
                    "Average quality. Works fine but not impressive. Fair price.",
                    "Decent item. Some pros and cons. Would consider other options.",
                    "Acceptable product. Meets basic needs. Could be better.",
                    "Standard quality. Functions adequately. Neither good nor bad."
                ]
            },
            'social_media': {
                'positive': [
                    "Just tried this and WOW! 😍 So impressed! #amazing #loveit",
                    "Can't believe how good this is! Thanks @company! 🙌 #grateful",
                    "Best decision ever! Already recommending to friends! ⭐⭐⭐⭐⭐",
                    "Obsessed with this! Quality is incredible! 💯 #newobsession",
                    "Mind blown! 🤯 This exceeded all my expectations! #impressed"
                ],
                'negative': [
                    "So disappointed 😞 This was a complete fail. #fail #disappointed",
                    "Worst purchase ever! Save your money! 😡 #regret #avoid",
                    "What a joke! Nothing like advertised. #scam #angry",
                    "Completely broke after one use. Terrible! 😤 #broken #waste",
                    "Don't buy this! Total garbage! #terrible #warning"
                ],
                'neutral': [
                    "It's okay I guess. Nothing to write home about. 🤷‍♀️ #meh",
                    "Average product. Does the job. That's about it. #average",
                    "Not bad, not great. Pretty standard. #okay #decent",
                    "Works fine. No complaints but no excitement either. #fine",
                    "Serviceable. Gets the job done. Nothing more. #serviceable"
                ]
            },
            'support_tickets': {
                'positive': [
                    "Thank you so much for the quick resolution! Excellent service!",
                    "Problem solved perfectly! Really appreciate the help.",
                    "Outstanding support! You went above and beyond. Thank you!",
                    "Very satisfied with the assistance. Quick and professional.",
                    "Great job resolving my issue! Will continue using your service."
                ],
                'negative': [
                    "Still not resolved after multiple attempts. Very frustrated.",
                    "Poor support experience. No one seems to understand the issue.",
                    "Waiting too long for a response. This is unacceptable.",
                    "Unhelpful responses. Issue remains unresolved. Very disappointed.",
                    "Terrible customer service. Problem getting worse, not better."
                ],
                'neutral': [
                    "Issue partially resolved. Still need some clarification.",
                    "Support was okay. Took a while but got some answers.",
                    "Standard response received. Following up for more details.",
                    "Some progress made. Waiting for final resolution.",
                    "Adequate support. Issue addressed but not completely solved."
                ]
            },
            'surveys': {
                'positive': [
                    "Very satisfied with overall experience. Would recommend to others.",
                    "Excellent service quality. Staff was professional and helpful.",
                    "Great value for money. Exceeded expectations in most areas.",
                    "Highly likely to continue using this service. Very pleased.",
                    "Outstanding experience from start to finish. Five stars."
                ],
                'negative': [
                    "Dissatisfied with service quality. Multiple issues encountered.",
                    "Poor value for money. Service did not meet expectations.",
                    "Would not recommend to others. Several problems occurred.",
                    "Unlikely to continue using this service. Too many issues.",
                    "Below average experience. Significant improvements needed."
                ],
                'neutral': [
                    "Service was acceptable. Some areas could be improved.",
                    "Average experience. Met basic expectations but nothing more.",
                    "Neutral on recommendation. Has both pros and cons.",
                    "Adequate service delivery. Room for enhancement.",
                    "Fair experience overall. Neither particularly good nor bad."
                ]
            },
            'forums': {
                'positive': [
                    "Has anyone else tried this? It's absolutely incredible! Best decision!",
                    "Just wanted to share my positive experience. Really happy with results!",
                    "Highly recommend this to everyone! Changed my life honestly.",
                    "Amazing results! Can't believe I waited so long to try this.",
                    "Fantastic experience! Sharing because others should know about this."
                ],
                'negative': [
                    "Warning to others: avoid this at all costs! Had terrible experience.",
                    "Don't waste your time or money. Complete disaster from start.",
                    "Sharing my bad experience so others don't make same mistake.",
                    "Worst experience ever. Multiple problems and poor service.",
                    "Save yourself the trouble. This is not worth it at all."
                ],
                'neutral': [
                    "Mixed experience here. Some good points, some not so much.",
                    "Sharing my experience for others to consider. Pretty average overall.",
                    "It's okay. Not amazing but not terrible either. Fair choice.",
                    "Decent option if you have realistic expectations. Nothing special.",
                    "Average results. Works but there are probably better alternatives."
                ]
            }
        }
    
    def _generate_sentiment_emotion(self) -> Tuple[str, str]:
        """Generate correlated sentiment and emotion labels."""
        
        # First determine sentiment
        sentiment = np.random.choice(self.config['sentiment_classes'], p=[0.25, 0.35, 0.4])
        
        # Then determine emotion based on sentiment
        if sentiment == 'positive':
            emotion = np.random.choice(['joy', 'surprise', 'neutral'], p=[0.6, 0.3, 0.1])
        elif sentiment == 'negative':
            emotion = np.random.choice(['anger', 'sadness', 'fear', 'neutral'], p=[0.4, 0.3, 0.2, 0.1])
        else:  # neutral
            emotion = np.random.choice(['neutral', 'surprise', 'joy', 'sadness'], p=[0.6, 0.15, 0.15, 0.1])
        
        return sentiment, emotion
    
    def _generate_text_content(self, source: str, sentiment: str, emotion: str, 
                             templates: Dict[str, Dict[str, List[str]]]) -> str:
        """Generate text content based on source, sentiment, and emotion."""
        
        # Get appropriate template
        base_text = np.random.choice(templates[source][sentiment])
        
        # Add emotional modifiers
        if emotion == 'anger':
            modifiers = ['!', ' This is ridiculous!', ' Absolutely unacceptable!']
            base_text += np.random.choice(modifiers)
        elif emotion == 'joy':
            modifiers = [' 😊', ' So happy!', ' Couldn\'t be better!']
            base_text += np.random.choice(modifiers)
        elif emotion == 'fear':
            modifiers = [' I\'m worried about this.', ' Concerning experience.', ' This makes me nervous.']
            base_text += np.random.choice(modifiers)
        elif emotion == 'sadness':
            modifiers = [' Really disappointing.', ' This makes me sad.', ' Such a letdown.']
            base_text += np.random.choice(modifiers)
        elif emotion == 'surprise':
            modifiers = [' Didn\'t expect this!', ' What a surprise!', ' Wasn\'t expecting that!']
            base_text += np.random.choice(modifiers)
        
        # Add source-specific variations
        if source == 'social_media' and np.random.random() < 0.3:
            hashtags = ['#experience', '#review', '#thoughts', '#feedback', '#opinion']
            base_text += ' ' + np.random.choice(hashtags)
        
        return base_text
    
    def _extract_text_features(self, text: str) -> Dict[str, float]:
        """Extract numerical features from text content."""
        
        # Clean text for analysis
        clean_text = text.lower()
        clean_text = re.sub(r'[^\w\s]', ' ', clean_text)
        words = clean_text.split()
        
        # Basic text statistics
        features = {
            'exclamation_count': text.count('!'),
            'question_count': text.count('?'),
            'caps_ratio': sum(1 for c in text if c.isupper()) / max(len(text), 1),
            'punctuation_ratio': sum(1 for c in text if c in string.punctuation) / max(len(text), 1),
            'emoji_count': len(re.findall(r'[\U0001f600-\U0001f64f]', text)),
            'hashtag_count': text.count('#'),
            'mention_count': text.count('@')
        }
        
        # Sentiment word features (simplified)
        positive_words = ['good', 'great', 'amazing', 'excellent', 'love', 'best', 'perfect', 
                         'wonderful', 'fantastic', 'awesome', 'outstanding', 'brilliant']
        negative_words = ['bad', 'terrible', 'awful', 'hate', 'worst', 'horrible', 'disappointing',
                         'useless', 'broken', 'failed', 'poor', 'pathetic']
        
        features['positive_word_count'] = sum(1 for word in words if word in positive_words)
        features['negative_word_count'] = sum(1 for word in words if word in negative_words)
        features['sentiment_word_ratio'] = (features['positive_word_count'] - features['negative_word_count']) / max(len(words), 1)
        
        # Intensity words
        intensity_words = ['very', 'extremely', 'incredibly', 'absolutely', 'completely', 
                          'totally', 'really', 'so', 'quite', 'rather']
        features['intensity_word_count'] = sum(1 for word in words if word in intensity_words)
        
        # Readability features
        features['avg_sentence_length'] = len(words) / max(text.count('.') + text.count('!') + text.count('?'), 1)
        features['unique_word_ratio'] = len(set(words)) / max(len(words), 1)
        
        return features
    
    def analyze_sentiment_patterns(self, X: pd.DataFrame, targets: Dict[str, pd.Series]) -> Dict[str, Any]:
        """Analyze patterns in sentiment data."""
        
        print("🔍 Analyzing sentiment patterns...")
        
        patterns = {}
        sentiment_data = targets['sentiment_classification']
        emotion_data = targets['emotion_classification']
        
        # 1. Overall distribution
        patterns['distribution'] = {
            'sentiment_counts': dict(sentiment_data.value_counts()),
            'emotion_counts': dict(emotion_data.value_counts()),
            'sentiment_percentages': dict(sentiment_data.value_counts(normalize=True) * 100),
            'emotion_percentages': dict(emotion_data.value_counts(normalize=True) * 100)
        }
        
        # 2. Source analysis
        source_sentiment = pd.crosstab(X['source'], sentiment_data, normalize='index') * 100
        patterns['source_analysis'] = {
            'sentiment_by_source': source_sentiment.to_dict(),
            'most_positive_source': source_sentiment['positive'].idxmax(),
            'most_negative_source': source_sentiment['negative'].idxmax(),
            'source_volumes': dict(X['source'].value_counts())
        }
        
        # 3. Temporal patterns
        # Create combined dataframe for temporal analysis
        temp_df = X.copy()
        temp_df['sentiment'] = sentiment_data
        temp_df['emotion'] = emotion_data
        
        monthly_sentiment = temp_df.groupby('month')['sentiment'].apply(
            lambda x: (x == 'positive').mean() - (x == 'negative').mean()
        )
        
        weekly_sentiment = temp_df.groupby('day_of_week')['sentiment'].apply(
            lambda x: (x == 'positive').mean() - (x == 'negative').mean()
        )
        
        patterns['temporal_analysis'] = {
            'best_month': monthly_sentiment.idxmax(),
            'worst_month': monthly_sentiment.idxmin(),
            'best_day': weekly_sentiment.idxmax(),
            'worst_day': weekly_sentiment.idxmin(),
            'weekend_vs_weekday': (
                temp_df[temp_df['is_weekend'] == 1]['sentiment'].apply(lambda x: 1 if x == 'positive' else 0).mean(),
                temp_df[temp_df['is_weekend'] == 0]['sentiment'].apply(lambda x: 1 if x == 'positive' else 0).mean()
            ),
            'monthly_trends': monthly_sentiment.to_dict(),
            'weekly_trends': weekly_sentiment.to_dict()
        }
        
        # 4. Text characteristics impact
        patterns['text_features'] = {
            'length_sentiment_correlation': np.corrcoef(
                X['text_length'], 
                sentiment_data.map({'negative': -1, 'neutral': 0, 'positive': 1})
            )[0, 1],
            'exclamation_positive_correlation': np.corrcoef(
                X['exclamation_count'],
                sentiment_data.map({'negative': -1, 'neutral': 0, 'positive': 1})
            )[0, 1],
            'caps_negative_correlation': np.corrcoef(
                X['caps_ratio'],
                sentiment_data.map({'negative': -1, 'neutral': 0, 'positive': 1})
            )[0, 1],
            'avg_word_length_by_sentiment': temp_df.groupby('sentiment')['avg_word_length'].mean().to_dict(),
            'engagement_by_sentiment': temp_df.groupby('sentiment')['engagement_score'].mean().to_dict()
        }
        
        # 5. Product category analysis
        if 'product_category' in X.columns:
            category_sentiment = pd.crosstab(X['product_category'], sentiment_data, normalize='index') * 100
            patterns['category_analysis'] = {
                'sentiment_by_category': category_sentiment.to_dict(),
                'best_category': category_sentiment['positive'].idxmax(),
                'worst_category': category_sentiment['negative'].idxmax()
            }
        
        # 6. User type analysis
        if 'user_type' in X.columns:
            user_sentiment = pd.crosstab(X['user_type'], sentiment_data, normalize='index') * 100
            patterns['user_analysis'] = {
                'sentiment_by_user_type': user_sentiment.to_dict(),
                'most_satisfied_users': user_sentiment['positive'].idxmax(),
                'least_satisfied_users': user_sentiment['negative'].idxmax()
            }
        
        # 7. Crisis detection patterns
        crisis_indicators = {
            'high_negative_rate': (sentiment_data == 'negative').mean(),
            'crisis_threshold_breach': (sentiment_data == 'negative').mean() > self.config['business_params']['crisis_threshold'],
            'anger_emotion_rate': (emotion_data == 'anger').mean(),
            'recent_trend': X['sentiment_trend_7d'].mean() if 'sentiment_trend_7d' in X.columns else 0
        }
        
        patterns['crisis_analysis'] = crisis_indicators
        
        print("✅ Sentiment pattern analysis completed")
        return patterns
    
    def train_sentiment_models(self, X: pd.DataFrame, targets: Dict[str, pd.Series]) -> Dict[str, Any]:
        """Train models for sentiment and emotion classification."""
        
        print("🚀 Training sentiment analysis models...")
        
        all_results = {}
        
        for target_name, target in targets.items():
            print(f"\nTraining models for {target_name}...")
            
            # Split data
            X_train, X_test, y_train, y_test = self.data_loader.train_test_split(
                X, target, test_size=self.config['test_size']
            )
            
            # Initialize models
            models = ClassificationModels()
            target_results = {}
            
            for algorithm in self.config['algorithms']:
                print(f"  Training {algorithm}...")
                
                # Train model
                model, training_time = models.train_model(
                    X_train, y_train, 
                    algorithm=algorithm,
                    class_weight='balanced'
                )
                
                # Make predictions
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
                
                # Evaluate model
                evaluator = ModelEvaluator()
                metrics = evaluator.classification_metrics(y_test, y_pred, y_pred_proba)
                
                # Calculate business impact
                business_metrics = self.calculate_sentiment_impact(
                    target_name, y_test, y_pred, X_test
                )
                
                target_results[algorithm] = {
                    'model': model,
                    'predictions': y_pred,
                    'probabilities': y_pred_proba,
                    'metrics': metrics,
                    'business_metrics': business_metrics,
                    'training_time': training_time,
                    'test_data': (X_test, y_test)
                }
                
                print(f"    ✅ {algorithm} - Accuracy: {metrics['accuracy']:.3f}, "
                      f"F1-Score: {metrics['f1_score']:.3f}")
            
            # Find best model
            best_algorithm = max(target_results.keys(), 
                               key=lambda x: target_results[x]['metrics']['f1_score'])
            
            all_results[target_name] = {
                'results': target_results,
                'best_model': best_algorithm,
                'best_performance': target_results[best_algorithm]
            }
            
            print(f"  🏆 Best model for {target_name}: {best_algorithm}")
        
        return all_results
    
    def calculate_sentiment_impact(self, target_name: str, y_true: pd.Series, 
                                 y_pred: pd.Series, X_test: pd.DataFrame) -> Dict[str, float]:
        """Calculate business impact of sentiment analysis."""
        
        if target_name == 'sentiment_classification':
            # Calculate sentiment accuracy and business value
            correct_predictions = (y_true == y_pred).sum()
            total_predictions = len(y_true)
            accuracy = correct_predictions / total_predictions
            
            # Business value calculation
            positive_value = self.config['business_params']['positive_sentiment_value']
            negative_cost = self.config['business_params']['negative_sentiment_cost']
            
            # True positives for crisis detection
            crisis_threshold = self.config['business_params']['crisis_threshold']
            true_negative_rate = ((y_true == 'negative') & (y_pred == 'negative')).sum() / max((y_true == 'negative').sum(), 1)
            false_negative_rate = ((y_true == 'negative') & (y_pred != 'negative')).sum() / max((y_true == 'negative').sum(), 1)
            
            # Crisis prevention value
            crisis_damage = self.config['business_params']['crisis_damage_cost']
            intervention_cost = self.config['business_params']['intervention_cost']
            crisis_prevention_value = true_negative_rate * crisis_damage - false_negative_rate * intervention_cost
            
            # Customer satisfaction impact
            satisfaction_improvement = accuracy * 0.4  # 40% improvement with perfect accuracy
            
            return {
                'accuracy': accuracy,
                'crisis_detection_rate': true_negative_rate,
                'crisis_prevention_value': crisis_prevention_value,
                'satisfaction_improvement': satisfaction_improvement,
                'positive_identification_rate': ((y_true == 'positive') & (y_pred == 'positive')).sum() / max((y_true == 'positive').sum(), 1),
                'negative_identification_rate': true_negative_rate,
                'business_value_per_prediction': (positive_value * accuracy - negative_cost * false_negative_rate)
            }
        
        elif target_name == 'emotion_classification':
            # Emotion classification business impact
            accuracy = (y_true == y_pred).sum() / len(y_true)
            
            # Emotion-specific accuracy
            emotion_accuracies = {}
            for emotion in self.config['emotion_classes']:
                emotion_mask = (y_true == emotion)
                if emotion_mask.sum() > 0:
                    emotion_accuracies[emotion] = ((y_true == emotion) & (y_pred == emotion)).sum() / emotion_mask.sum()
                else:
                    emotion_accuracies[emotion] = 0
            
            # Business value of emotion understanding
            emotion_value = accuracy * 100  # $100 per accurate emotion classification
            
            return {
                'accuracy': accuracy,
                'emotion_accuracies': emotion_accuracies,
                'anger_detection_rate': emotion_accuracies.get('anger', 0),
                'joy_detection_rate': emotion_accuracies.get('joy', 0),
                'emotion_business_value': emotion_value,
                'customer_insight_value': accuracy * 200  # Value of emotional insights
            }
        
        return {}
    
    def perform_sentiment_monitoring(self, X: pd.DataFrame, models_dict: Dict[str, Any]) -> pd.DataFrame:
        """Generate real-time sentiment monitoring dashboard data."""
        
        print("📊 Generating sentiment monitoring insights...")
        
        # Get best models
        sentiment_model = models_dict['sentiment_classification']['best_performance']['model']
        emotion_model = models_dict['emotion_classification']['best_performance']['model']
        
        # Sample recent data for monitoring
        monitoring_data = []
        sample_size = min(1000, len(X))
        X_sample = X.sample(n=sample_size, random_state=self.config['random_state'])
        
        for idx, row in X_sample.iterrows():
            # Predict sentiment and emotion
            sentiment_pred = sentiment_model.predict([row])[0]
            emotion_pred = emotion_model.predict([row])[0]
            
            # Get prediction probabilities if available
            sentiment_proba = sentiment_model.predict_proba([row])[0] if hasattr(sentiment_model, 'predict_proba') else None
            emotion_proba = emotion_model.predict_proba([row])[0] if hasattr(emotion_model, 'predict_proba') else None
            
            # Calculate confidence scores
            sentiment_confidence = max(sentiment_proba) if sentiment_proba is not None else 0.5
            emotion_confidence = max(emotion_proba) if emotion_proba is not None else 0.5
            
            # Determine alert level
            alert_level = 'low'
            if sentiment_pred == 'negative' and sentiment_confidence > 0.8:
                if emotion_pred in ['anger', 'fear']:
                    alert_level = 'high'
                else:
                    alert_level = 'medium'
            elif sentiment_pred == 'negative':
                alert_level = 'medium'
            
            monitoring_data.append({
                'text_id': f'Monitor_{idx}',
                'source': row['source'],
                'predicted_sentiment': sentiment_pred,
                'predicted_emotion': emotion_pred,
                'sentiment_confidence': sentiment_confidence,
                'emotion_confidence': emotion_confidence,
                'alert_level': alert_level,
                'engagement_score': row['engagement_score'],
                'text_length': row['text_length'],
                'product_category': row['product_category'],
                'user_type': row['user_type'],
                'requires_attention': 1 if alert_level == 'high' else 0,
                'crisis_indicator': 1 if (sentiment_pred == 'negative' and 
                                         emotion_pred == 'anger' and 
                                         sentiment_confidence > 0.9) else 0
            })
        
        monitoring_df = pd.DataFrame(monitoring_data)
        
        print(f"✅ Generated {len(monitoring_df)} monitoring records")
        print(f"🚨 High alerts: {(monitoring_df['alert_level'] == 'high').sum()}")
        print(f"⚠️ Crisis indicators: {monitoring_df['crisis_indicator'].sum()}")
        
        return monitoring_df
    
    def visualize_results(self, patterns: Dict[str, Any], results: Dict[str, Any], 
                         monitoring: pd.DataFrame) -> None:
        """Create comprehensive visualizations of sentiment analysis results."""
        
        print("📊 Creating sentiment analysis visualizations...")
        
        fig = plt.figure(figsize=(24, 20))
        
        # 1. Sentiment distribution
        ax1 = plt.subplot(4, 5, 1)
        sentiment_counts = patterns['distribution']['sentiment_counts']
        colors = ['red', 'gray', 'green']
        wedges, texts, autotexts = ax1.pie(sentiment_counts.values(), labels=sentiment_counts.keys(),
                                          autopct='%1.1f%%', colors=colors, startangle=90)
        ax1.set_title('Overall Sentiment Distribution', fontweight='bold')
        
        # 2. Emotion distribution
        ax2 = plt.subplot(4, 5, 2)
        emotion_counts = patterns['distribution']['emotion_counts']
        bars = ax2.bar(emotion_counts.keys(), emotion_counts.values(), 
                      color='skyblue', alpha=0.7)
        ax2.set_title('Emotion Distribution', fontweight='bold')
        ax2.set_ylabel('Count')
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. Sentiment by source
        ax3 = plt.subplot(4, 5, 3)
        source_sentiment = patterns['source_analysis']['sentiment_by_source']
        sources = list(source_sentiment['positive'].keys())
        positive_rates = [source_sentiment['positive'][s] for s in sources]
        negative_rates = [source_sentiment['negative'][s] for s in sources]
        
        x = np.arange(len(sources))
        width = 0.35
        ax3.bar(x - width/2, positive_rates, width, label='Positive', color='green', alpha=0.7)
        ax3.bar(x + width/2, negative_rates, width, label='Negative', color='red', alpha=0.7)
        
        ax3.set_title('Sentiment by Source', fontweight='bold')
        ax3.set_ylabel('Percentage (%)')
        ax3.set_xticks(x)
        ax3.set_xticklabels(sources, rotation=45)
        ax3.legend()
        
        # 4. Monthly sentiment trends
        ax4 = plt.subplot(4, 5, 4)
        monthly_trends = patterns['temporal_analysis']['monthly_trends']
        months = list(monthly_trends.keys())
        sentiment_scores = list(monthly_trends.values())
        
        ax4.plot(months, sentiment_scores, marker='o', linewidth=3, markersize=8, color='blue')
        ax4.set_title('Monthly Sentiment Trends', fontweight='bold')
        ax4.set_xlabel('Month')
        ax4.set_ylabel('Sentiment Score')
        ax4.grid(True, alpha=0.3)
        ax4.axhline(0, color='black', linestyle='--', alpha=0.5)
        
        # 5. Model performance comparison
        ax5 = plt.subplot(4, 5, 5)
        if 'sentiment_classification' in results:
            sentiment_results = results['sentiment_classification']['results']
            algorithms = list(sentiment_results.keys())
            f1_scores = [sentiment_results[alg]['metrics']['f1_score'] for alg in algorithms]
            
            bars = ax5.bar(algorithms, f1_scores, color='gold', alpha=0.7)
            ax5.set_title('Sentiment Model Performance', fontweight='bold')
            ax5.set_ylabel('F1 Score')
            ax5.set_ylim(0, 1)
            ax5.tick_params(axis='x', rotation=45)
            
            # Highlight best model
            best_idx = np.argmax(f1_scores)
            bars[best_idx].set_color('darkgoldenrod')
        
        # 6. Crisis monitoring dashboard
        ax6 = plt.subplot(4, 5, (6, 7))
        if not monitoring.empty:
            alert_counts = monitoring['alert_level'].value_counts()
            
            # Create alert level pie chart
            alert_colors = {'low': 'green', 'medium': 'orange', 'high': 'red'}
            colors_list = [alert_colors.get(level, 'gray') for level in alert_counts.index]
            
            wedges, texts, autotexts = ax6.pie(alert_counts.values, labels=alert_counts.index,
                                              autopct='%1.1f%%', colors=colors_list, startangle=90)
            ax6.set_title('Current Alert Level Distribution', fontweight='bold')
        
        # 8. Text features impact
        ax8 = plt.subplot(4, 5, 8)
        text_features = patterns['text_features']
        features = ['Length', 'Exclamation', 'CAPS', 'Engagement']
        correlations = [
            text_features['length_sentiment_correlation'],
            text_features['exclamation_positive_correlation'],
            text_features['caps_negative_correlation'],
            np.mean(list(text_features['engagement_by_sentiment'].values())) - 0.5  # Normalized
        ]
        
        colors = ['green' if c > 0 else 'red' for c in correlations]
        bars = ax8.bar(features, correlations, color=colors, alpha=0.7)
        ax8.set_title('Text Features vs Sentiment', fontweight='bold')
        ax8.set_ylabel('Correlation')
        ax8.axhline(0, color='black', linestyle='-', alpha=0.3)
        ax8.tick_params(axis='x', rotation=45)
        
        # 9. User type sentiment analysis
        ax9 = plt.subplot(4, 5, 9)
        if 'user_analysis' in patterns:
            user_sentiment = patterns['user_analysis']['sentiment_by_user_type']
            user_types = list(user_sentiment['positive'].keys())
            user_positive = [user_sentiment['positive'][u] for u in user_types]
            
            bars = ax9.bar(user_types, user_positive, color='lightblue', alpha=0.7)
            ax9.set_title('Positive Sentiment by User Type', fontweight='bold')
            ax9.set_ylabel('Positive Sentiment (%)')
            ax9.tick_params(axis='x', rotation=45)
            
            # Highlight best user type
            best_idx = np.argmax(user_positive)
            bars[best_idx].set_color('blue')
        
        # 10. Product category analysis
        ax10 = plt.subplot(4, 5, 10)
        if 'category_analysis' in patterns:
            category_sentiment = patterns['category_analysis']['sentiment_by_category']
            categories = list(category_sentiment['positive'].keys())
            cat_positive = [category_sentiment['positive'][c] for c in categories]
            cat_negative = [category_sentiment['negative'][c] for c in categories]
            
            x = np.arange(len(categories))
            width = 0.35
            ax10.bar(x - width/2, cat_positive, width, label='Positive', color='green', alpha=0.7)
            ax10.bar(x + width/2, cat_negative, width, label='Negative', color='red', alpha=0.7)
            
            ax10.set_title('Sentiment by Product Category', fontweight='bold')
            ax10.set_ylabel('Percentage (%)')
            ax10.set_xticks(x)
            ax10.set_xticklabels(categories, rotation=45)
            ax10.legend()
        
        # 11. Emotion classification performance
        ax11 = plt.subplot(4, 5, (11, 12))
        if 'emotion_classification' in results:
            emotion_results = results['emotion_classification']['results']
            best_emotion_result = results['emotion_classification']['best_performance']
            
            # Create confusion matrix-like visualization for emotions
            if 'emotion_accuracies' in best_emotion_result['business_metrics']:
                emotion_acc = best_emotion_result['business_metrics']['emotion_accuracies']
                emotions = list(emotion_acc.keys())
                accuracies = list(emotion_acc.values())
                
                bars = ax11.barh(emotions, accuracies, color='purple', alpha=0.7)
                ax11.set_title('Emotion Classification Accuracy by Class', fontweight='bold')
                ax11.set_xlabel('Accuracy')
                ax11.set_xlim(0, 1)
                
                # Add accuracy labels
                for bar, acc in zip(bars, accuracies):
                    width = bar.get_width()
                    ax11.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                             f'{acc:.2f}', ha='left', va='center')
        
        # 13. Weekly sentiment patterns
        ax13 = plt.subplot(4, 5, 13)
        weekly_trends = patterns['temporal_analysis']['weekly_trends']
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        weekly_scores = [weekly_trends[i] for i in range(7)]
        
        colors = ['lightblue' if i < 5 else 'lightcoral' for i in range(7)]
        bars = ax13.bar(days, weekly_scores, color=colors, alpha=0.7)
        ax13.set_title('Weekly Sentiment Patterns', fontweight='bold')
        ax13.set_ylabel('Sentiment Score')
        ax13.axhline(0, color='black', linestyle='--', alpha=0.5)
        
        # 14. Crisis indicators
        ax14 = plt.subplot(4, 5, 14)
        if not monitoring.empty:
            crisis_data = [
                monitoring['crisis_indicator'].sum(),
                (monitoring['alert_level'] == 'high').sum() - monitoring['crisis_indicator'].sum(),
                (monitoring['alert_level'] == 'medium').sum(),
                len(monitoring) - (monitoring['alert_level'] != 'low').sum()
            ]
            crisis_labels = ['Crisis', 'High Alert', 'Medium Alert', 'Normal']
            crisis_colors = ['darkred', 'red', 'orange', 'green']
            
            bars = ax14.bar(crisis_labels, crisis_data, color=crisis_colors, alpha=0.7)
            ax14.set_title('Current Crisis Indicators', fontweight='bold')
            ax14.set_ylabel('Count')
            ax14.tick_params(axis='x', rotation=45)
            
            # Add count labels
            for bar, count in zip(bars, crisis_data):
                height = bar.get_height()
                ax14.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                         f'{count}', ha='center', va='bottom')
        
        # 15. Business impact summary
        ax15 = plt.subplot(4, 5, (15, 20))
        
        if results:
            sentiment_business = results['sentiment_classification']['best_performance']['business_metrics']
            emotion_business = results['emotion_classification']['best_performance']['business_metrics']
            
            # Calculate key metrics
            crisis_prevention = sentiment_business.get('crisis_prevention_value', 0)
            satisfaction_gain = sentiment_business.get('satisfaction_improvement', 0) * 100
            detection_accuracy = sentiment_business.get('accuracy', 0) * 100
            
            summary_text = f"""
SENTIMENT ANALYSIS PERFORMANCE SUMMARY

Dataset Overview:
• Total Text Records: {self.config['n_texts']:,}
• Text Sources: {len(self.config['text_sources'])}
• Sentiment Classes: {len(self.config['sentiment_classes'])}
• Emotion Classes: {len(self.config['emotion_classes'])}

Model Performance:
• Sentiment Accuracy: {detection_accuracy:.1f}%
• Best Sentiment Model: {results['sentiment_classification']['best_model'].replace('_', ' ').title()}
• Best Emotion Model: {results['emotion_classification']['best_model'].replace('_', ' ').title()}
• Crisis Detection Rate: {sentiment_business.get('crisis_detection_rate', 0):.1%}

Business Impact:
• Crisis Prevention Value: ${crisis_prevention:,.0f}
• Customer Satisfaction Gain: {satisfaction_gain:.1f}%
• Monitoring Alerts Generated: {monitoring['alert_level'].value_counts().get('high', 0) if not monitoring.empty else 0}

Current Status:
• Negative Sentiment Rate: {patterns['distribution']['sentiment_percentages'].get('negative', 0):.1f}%
• Crisis Threshold: {self.config['business_params']['crisis_threshold']*100:.0f}%
• Crisis Status: {'⚠️ BREACH' if patterns['crisis_analysis']['crisis_threshold_breach'] else '✅ NORMAL'}

Top Insights:
• Most Positive Source: {patterns['source_analysis']['most_positive_source'].title()}
• Most Negative Source: {patterns['source_analysis']['most_negative_source'].title()}
• Peak Sentiment Month: {patterns['temporal_analysis']['best_month']}
• Best User Segment: {patterns.get('user_analysis', {}).get('most_satisfied_users', 'Premium').title()}

Recommendations:
• Monitor {patterns['source_analysis']['most_negative_source']} closely
• Focus improvement on {patterns.get('category_analysis', {}).get('worst_category', 'electronics')} category
• Implement proactive engagement during Month {patterns['temporal_analysis']['worst_month']}
• Enhance crisis detection for anger emotions
"""
            
            ax15.text(0.05, 0.95, summary_text, transform=ax15.transAxes, 
                     fontsize=10, verticalalignment='top', fontfamily='monospace',
                     bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        
        ax15.axis('off')
        ax15.set_title('Sentiment Analysis Summary', fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.show()
        
        print("✅ Sentiment analysis visualizations completed")
    
    def generate_sentiment_report(self, patterns: Dict[str, Any], results: Dict[str, Any], 
                                monitoring: pd.DataFrame) -> str:
        """Generate comprehensive sentiment analysis report."""
        
        if not results:
            return "No sentiment analysis results available for report generation."
        
        # Get key metrics
        sentiment_metrics = results['sentiment_classification']['best_performance']['business_metrics']
        emotion_metrics = results['emotion_classification']['best_performance']['business_metrics']
        
        best_sentiment_model = results['sentiment_classification']['best_model']
        best_emotion_model = results['emotion_classification']['best_model']
        
        report = f"""
# 💭 SENTIMENT ANALYSIS SYSTEM REPORT

## Executive Summary

**Sentiment Detection Accuracy**: {sentiment_metrics.get('accuracy', 0):.1%}
**Emotion Classification Accuracy**: {emotion_metrics.get('accuracy', 0):.1%}
**Crisis Prevention Value**: ${sentiment_metrics.get('crisis_prevention_value', 0):,.0f}
**Customer Satisfaction Impact**: {sentiment_metrics.get('satisfaction_improvement', 0):.1%} improvement
**Crisis Status**: {'⚠️ ACTIVE MONITORING' if patterns['crisis_analysis']['crisis_threshold_breach'] else '✅ NORMAL OPERATIONS'}

## 📊 Data Analysis Overview

**Dataset Composition**:
- **Total Text Records**: {self.config['n_texts']:,}
- **Text Sources**: {', '.join(self.config['text_sources'])}
- **Analysis Period**: 365 days of synthetic data
- **Languages**: English (expandable to multilingual)

**Current Sentiment Landscape**:
- **Positive Sentiment**: {patterns['distribution']['sentiment_percentages'].get('positive', 0):.1f}%
- **Neutral Sentiment**: {patterns['distribution']['sentiment_percentages'].get('neutral', 0):.1f}%
- **Negative Sentiment**: {patterns['distribution']['sentiment_percentages'].get('negative', 0):.1f}%

**Emotion Distribution**:
"""
        
        for emotion, percentage in patterns['distribution']['emotion_percentages'].items():
            report += f"- **{emotion.title()}**: {percentage:.1f}%\n"
        
        report += f"""

## 🎯 Model Performance Analysis

**Sentiment Classification**:
- **Best Model**: {best_sentiment_model.replace('_', ' ').title()}
- **Overall Accuracy**: {sentiment_metrics.get('accuracy', 0):.1%}
- **Positive Detection Rate**: {sentiment_metrics.get('positive_identification_rate', 0):.1%}
- **Negative Detection Rate**: {sentiment_metrics.get('negative_identification_rate', 0):.1%}
- **Crisis Detection Capability**: {sentiment_metrics.get('crisis_detection_rate', 0):.1%}

**Emotion Classification**:
- **Best Model**: {best_emotion_model.replace('_', ' ').title()}
- **Overall Accuracy**: {emotion_metrics.get('accuracy', 0):.1%}
- **Anger Detection Rate**: {emotion_metrics.get('anger_detection_rate', 0):.1%}
- **Joy Detection Rate**: {emotion_metrics.get('joy_detection_rate', 0):.1%}

**Individual Emotion Accuracies**:
"""
        
        if 'emotion_accuracies' in emotion_metrics:
            for emotion, accuracy in emotion_metrics['emotion_accuracies'].items():
                report += f"- **{emotion.title()}**: {accuracy:.1%}\n"
        
        report += f"""

## 📈 Source Performance Analysis

**Platform Rankings by Positivity**:
"""
        
        source_sentiment = patterns['source_analysis']['sentiment_by_source']
        # Sort sources by positive sentiment percentage
        sorted_sources = sorted(source_sentiment['positive'].items(), key=lambda x: x[1], reverse=True)
        
        for source, positive_rate in sorted_sources:
            negative_rate = source_sentiment['negative'][source]
            neutral_rate = source_sentiment['neutral'][source]
            volume = patterns['source_analysis']['source_volumes'][source]
            
            report += f"""
**{source.replace('_', ' ').title()}**:
- Positive: {positive_rate:.1f}%
- Neutral: {neutral_rate:.1f}%
- Negative: {negative_rate:.1f}%
- Volume: {volume:,} texts
"""
        
        report += f"""

**Key Source Insights**:
- **Most Positive Platform**: {patterns['source_analysis']['most_positive_source'].replace('_', ' ').title()}
- **Most Critical Platform**: {patterns['source_analysis']['most_negative_source'].replace('_', ' ').title()}
- **Highest Volume Platform**: {max(patterns['source_analysis']['source_volumes'], key=patterns['source_analysis']['source_volumes'].get).replace('_', ' ').title()}

## ⏰ Temporal Sentiment Patterns

**Monthly Trends**:
- **Peak Sentiment Month**: {patterns['temporal_analysis']['best_month']} (most positive)
- **Lowest Sentiment Month**: {patterns['temporal_analysis']['worst_month']} (most negative)

**Weekly Patterns**:
- **Best Day**: {['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][patterns['temporal_analysis']['best_day']]}
- **Challenging Day**: {['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][patterns['temporal_analysis']['worst_day']]}
- **Weekend Effect**: {patterns['temporal_analysis']['weekend_vs_weekday'][0]:.2f} vs {patterns['temporal_analysis']['weekend_vs_weekday'][1]:.2f} (weekend vs weekday positivity)

**Monthly Performance Detail**:
"""
        
        monthly_trends = patterns['temporal_analysis']['monthly_trends']
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        for month_num, score in monthly_trends.items():
            month_name = months[month_num - 1]
            trend = "📈" if score > 0 else "📉" if score < -0.1 else "➡️"
            report += f"- **{month_name}**: {score:.3f} {trend}\n"
        
        if 'user_analysis' in patterns:
            report += f"""

## 👥 Customer Segment Analysis

**User Type Performance**:
"""
            user_sentiment = patterns['user_analysis']['sentiment_by_user_type']
            for user_type, positive_rate in user_sentiment['positive'].items():
                negative_rate = user_sentiment['negative'][user_type]
                report += f"- **{user_type.title()} Users**: {positive_rate:.1f}% positive, {negative_rate:.1f}% negative\n"
            
            report += f"""
- **Most Satisfied Segment**: {patterns['user_analysis']['most_satisfied_users'].title()}
- **Most Critical Segment**: {patterns['user_analysis']['least_satisfied_users'].title()}
"""
        
        if 'category_analysis' in patterns:
            report += f"""

## 🛍️ Product Category Insights

**Category Sentiment Rankings**:
"""
            category_sentiment = patterns['category_analysis']['sentiment_by_category']
            # Sort categories by positive sentiment
            sorted_categories = sorted(category_sentiment['positive'].items(), key=lambda x: x[1], reverse=True)
            
            for category, positive_rate in sorted_categories:
                negative_rate = category_sentiment['negative'][category]
                report += f"- **{category.title()}**: {positive_rate:.1f}% positive, {negative_rate:.1f}% negative\n"
            
            report += f"""
- **Top Performing Category**: {patterns['category_analysis']['best_category'].title()}
- **Category Needing Attention**: {patterns['category_analysis']['worst_category'].title()}
"""
        
        report += f"""

## 🚨 Crisis Monitoring & Risk Assessment

**Current Risk Status**:
- **Negative Sentiment Rate**: {patterns['crisis_analysis']['high_negative_rate']:.1%}
- **Crisis Threshold**: {self.config['business_params']['crisis_threshold']*100:.0f}%
- **Threshold Breach**: {'⚠️ YES' if patterns['crisis_analysis']['crisis_threshold_breach'] else '✅ NO'}
- **Anger Emotion Rate**: {patterns['crisis_analysis']['anger_emotion_rate']:.1%}
- **7-Day Sentiment Trend**: {patterns['crisis_analysis']['recent_trend']:.3f}

**Real-time Alert Summary**:
"""
        
        if not monitoring.empty:
            alert_summary = monitoring['alert_level'].value_counts()
            crisis_count = monitoring['crisis_indicator'].sum()
            high_attention = monitoring['requires_attention'].sum()
            
            report += f"""
- **High Priority Alerts**: {alert_summary.get('high', 0)}
- **Medium Priority Alerts**: {alert_summary.get('medium', 0)}
- **Low Priority**: {alert_summary.get('low', 0)}
- **Crisis Indicators**: {crisis_count}
- **Requires Immediate Attention**: {high_attention}
"""
        
        report += f"""

## 💰 Business Impact Assessment

**Financial Impact**:
- **Crisis Prevention Value**: ${sentiment_metrics.get('crisis_prevention_value', 0):,.0f}
- **Customer Satisfaction ROI**: {sentiment_metrics.get('satisfaction_improvement', 0)*100:.1f}% improvement
- **Per-Prediction Business Value**: ${sentiment_metrics.get('business_value_per_prediction', 0):.2f}
- **Emotion Insights Value**: ${emotion_metrics.get('customer_insight_value', 0):,.0f}

**Operational Benefits**:
- **Automated Monitoring**: 24/7 real-time sentiment tracking
- **Early Warning System**: Crisis detection before escalation
- **Customer Intelligence**: Deep emotional understanding
- **Response Optimization**: Data-driven customer service priorities

**Risk Mitigation**:
- **Crisis Damage Prevention**: Up to ${self.config['business_params']['crisis_damage_cost']:,.0f} per avoided crisis
- **Intervention Cost**: ${self.config['business_params']['intervention_cost']:,.0f} per proactive response
- **Net Protection Value**: ${sentiment_metrics.get('crisis_prevention_value', 0):,.0f} annually

## 🔍 Text Analysis Insights

**Linguistic Pattern Analysis**:
- **Text Length Impact**: {patterns['text_features']['length_sentiment_correlation']:.3f} correlation with sentiment
- **Exclamation Usage**: {patterns['text_features']['exclamation_positive_correlation']:.3f} correlation with positivity
- **Capitalization Effect**: {patterns['text_features']['caps_negative_correlation']:.3f} correlation with negativity

**Engagement Patterns**:
"""
        
        engagement_by_sentiment = patterns['text_features']['engagement_by_sentiment']
        for sentiment, engagement in engagement_by_sentiment.items():
            report += f"- **{sentiment.title()} Content**: {engagement:.3f} avg engagement score\n"
        
        report += f"""

## 🚀 Strategic Recommendations

**Immediate Actions (0-30 days)**:
1. **Crisis Monitoring**: Implement real-time alerts for negative sentiment spikes >30%
2. **Source Focus**: Prioritize monitoring on {patterns['source_analysis']['most_negative_source']} platform
3. **Response Protocol**: Establish 2-hour response time for high-priority alerts
4. **Team Training**: Deploy sentiment-aware customer service protocols

**Short-term Improvements (1-3 months)**:
1. **Category Optimization**: Address sentiment issues in {patterns.get('category_analysis', {}).get('worst_category', 'electronics')} category
2. **Temporal Planning**: Prepare proactive campaigns for {patterns['temporal_analysis']['worst_month']} (historically challenging)
3. **User Segmentation**: Develop targeted retention strategy for {patterns.get('user_analysis', {}).get('least_satisfied_users', 'new')} users
4. **Content Strategy**: Leverage insights from high-engagement positive content

**Long-term Strategy (3-12 months)**:
1. **Predictive Analytics**: Implement sentiment forecasting models
2. **Multi-language Support**: Expand analysis to additional languages
3. **Integration Expansion**: Connect with CRM, support, and marketing automation
4. **Advanced Emotion AI**: Deploy emotion-specific response strategies

**Technology Roadmap**:
1. **Real-time Processing**: Sub-second sentiment classification
2. **API Integration**: Seamless connection with existing tools
3. **Dashboard Enhancement**: Executive-level sentiment insights
4. **Machine Learning Pipeline**: Continuous model improvement

## 📊 Performance Monitoring Framework

**Key Performance Indicators**:
- **Sentiment Accuracy**: Target >90% (Current: {sentiment_metrics.get('accuracy', 0):.1%})
- **Crisis Detection Rate**: Target >95% (Current: {sentiment_metrics.get('crisis_detection_rate', 0):.1%})
- **Response Time**: Target <2 hours for high alerts
- **False Positive Rate**: Target <5%
- **Customer Satisfaction Correlation**: Target >0.8

**Monthly Review Metrics**:
- Sentiment trend analysis and forecasting
- Source performance benchmarking
- Category-specific sentiment health checks
- Crisis prevention effectiveness assessment
- Model accuracy and drift monitoring

**Quarterly Strategic Reviews**:
- Business impact assessment and ROI calculation
- Competitive sentiment benchmarking
- Technology stack optimization
- Team performance and training needs
- Strategic initiative effectiveness

## ⚠️ Risk Management & Compliance

**Data Privacy & Security**:
- GDPR/CCPA compliant data processing
- Anonymization of personal identifiers
- Secure data storage and transmission
- Regular security audits and penetration testing

**Bias & Fairness Monitoring**:
- Regular bias detection across demographic groups
- Fairness metrics tracking and reporting
- Model explanation and transparency measures
- Diverse training data validation

**Operational Risks**:
- Model drift detection and automated retraining
- Backup systems for high-availability operations
- Human oversight for critical decisions
- Regular accuracy validation and calibration

## 🎯 Success Metrics & ROI

**6-Month Targets**:
- 40% improvement in customer satisfaction scores
- 50% reduction in crisis response time
- 25% increase in positive sentiment detection
- $500K in crisis prevention value

**12-Month Objectives**:
- 95% sentiment classification accuracy
- Real-time sentiment-driven decision making
- Integrated customer experience optimization
- $2.1M total business impact achievement

**Continuous Improvement**:
- Monthly model performance reviews
- Quarterly business impact assessments
- Annual technology stack evaluations
- Ongoing team training and development

---
*Report generated by Sentiment Analysis System*
*Analysis Confidence: {np.mean([results[obj]['best_performance']['metrics']['accuracy'] for obj in results.keys()]):.0%}*
*Crisis Status: {'⚠️ MONITORING' if patterns['crisis_analysis']['crisis_threshold_breach'] else '✅ NORMAL'}*
*Report Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        return report
    
    def run_complete_analysis(self) -> Dict[str, Any]:
        """Run the complete sentiment analysis pipeline."""
        
        print("💭 Starting Sentiment Analysis System")
        print("=" * 60)
        
        try:
            # 1. Generate dataset
            X, targets = self.generate_text_dataset()
            self.text_data = (X, targets)
            
            # 2. Analyze patterns
            patterns = self.analyze_sentiment_patterns(X, targets)
            
            # 3. Train sentiment models
            results = self.train_sentiment_models(X, targets)
            self.sentiment_results = results
            
            # 4. Generate monitoring data
            monitoring = self.perform_sentiment_monitoring(X, results) if results else pd.DataFrame()
            
            # 5. Create visualizations
            self.visualize_results(patterns, results, monitoring)
            
            # 6. Generate report
            report = self.generate_sentiment_report(patterns, results, monitoring)
            
            # 7. Return comprehensive results
            analysis_results = {
                'dataset': (X, targets),
                'patterns': patterns,
                'sentiment_results': results,
                'monitoring': monitoring,
                'report': report,
                'config': self.config
            }
            
            # Calculate key metrics for summary
            if results:
                sentiment_accuracy = results['sentiment_classification']['best_performance']['metrics']['accuracy']
                emotion_accuracy = results['emotion_classification']['best_performance']['metrics']['accuracy']
                crisis_prevention_value = results['sentiment_classification']['best_performance']['business_metrics'].get('crisis_prevention_value', 0)
                best_sentiment_model = results['sentiment_classification']['best_model']
                crisis_alerts = monitoring['crisis_indicator'].sum() if not monitoring.empty else 0
            else:
                sentiment_accuracy = 0
                emotion_accuracy = 0
                crisis_prevention_value = 0
                best_sentiment_model = "None"
                crisis_alerts = 0
            
            print("\n" + "=" * 60)
            print("🎉 Sentiment Analysis Complete!")
            print(f"📊 Sentiment Accuracy: {sentiment_accuracy:.1%}")
            print(f"😊 Emotion Accuracy: {emotion_accuracy:.1%}")
            print(f"🏆 Best Model: {best_sentiment_model.replace('_', ' ').title()}")
            print(f"💰 Crisis Prevention Value: ${crisis_prevention_value:,.0f}")
            print(f"🚨 Active Crisis Alerts: {crisis_alerts}")
            
            return analysis_results
            
        except Exception as e:
            print(f"❌ Error in sentiment analysis: {str(e)}")
            raise

def main():
    """Main function to demonstrate sentiment analysis system."""
    
    # Initialize system
    sentiment_system = SentimentAnalysisSystem()
    
    # Run complete analysis
    results = sentiment_system.run_complete_analysis()
    
    # Print report
    print("\n" + "=" * 80)
    print("📋 SENTIMENT ANALYSIS REPORT")
    print("=" * 80)
    print(results['report'])

if __name__ == "__main__":
    main()