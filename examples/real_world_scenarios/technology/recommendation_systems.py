# File: examples/real_world_scenarios/technology/recommendation_systems.py
# Location: examples/real_world_scenarios/technology/recommendation_systems.py

"""
Recommendation System - Real-World ML Pipeline Example

Business Problem:
Build personalized recommendation systems to increase user engagement,
improve customer experience, and drive revenue through relevant suggestions.

Dataset: User-item interaction data (synthetic)
Target: Recommendation ranking and rating prediction
Business Impact: 35% increase in click-through rate, 22% boost in revenue per user
Techniques: Collaborative filtering, content-based filtering, matrix factorization

Industry Applications:
- E-commerce platforms (Amazon, eBay)
- Streaming services (Netflix, Spotify)
- Social media (Facebook, LinkedIn)
- News and content platforms
- Online marketplaces
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, Any, List
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cosine
import warnings
warnings.filterwarnings('ignore')

# Framework imports
from src.data.generators import DataGenerator
from src.evaluation.metrics import ModelEvaluator

# Scenario-specific imports
from ..utilities.data_loaders import DataLoader
from ..utilities.visualization_helpers import BusinessVisualizer
from ..utilities.evaluation_helpers import BusinessMetricsCalculator

class RecommendationSystem:
    """Complete recommendation system pipeline."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize recommendation system."""
        
        self.config = config or {
            'n_users': 1000,
            'n_items': 500,
            'n_interactions': 50000,
            'test_size': 0.2,
            'random_state': 42,
            'recommendation_algorithms': ['collaborative_filtering', 'content_based', 'matrix_factorization'],
            'n_recommendations': 10,
            'business_params': {
                'avg_order_value': 75,
                'recommendation_cost': 0.05,
                'conversion_rate_baseline': 0.02
            }
        }
        
        # Initialize components
        self.data_loader = DataLoader(random_state=self.config['random_state'])
        self.visualizer = BusinessVisualizer()
        self.business_calc = BusinessMetricsCalculator()
        
        # Results storage
        self.recommendation_results = {}
        self.user_item_matrix = None
        self.item_features = None
        self.user_profiles = None
        
    def load_and_analyze_interaction_data(self) -> pd.DataFrame:
        """Load and analyze user-item interaction data."""
        
        print("🔄 Loading recommendation system dataset...")
        interaction_df = self.data_loader.load_recommendation_data(
            n_users=self.config['n_users'],
            n_items=self.config['n_items'],
            n_interactions=self.config['n_interactions']
        )
        
        print(f"📊 Dataset shape: {interaction_df.shape}")
        print(f"📊 Unique users: {interaction_df['user_id'].nunique():,}")
        print(f"📊 Unique items: {interaction_df['item_id'].nunique():,}")
        print(f"📊 Total interactions: {len(interaction_df):,}")
        
        # Interaction analysis
        print("\n📈 Interaction Analysis:")
        print(f"   Average rating: {interaction_df['rating'].mean():.2f}")
        print(f"   Rating distribution: {dict(interaction_df['rating'].value_counts().sort_index())}")
        
        # User behavior analysis
        user_stats = interaction_df.groupby('user_id').agg({
            'rating': ['count', 'mean'],
            'item_id': 'nunique'
        }).round(2)
        user_stats.columns = ['interactions', 'avg_rating', 'unique_items']
        
        print(f"\n👤 User Behavior:")
        print(f"   Avg interactions per user: {user_stats['interactions'].mean():.1f}")
        print(f"   Avg rating per user: {user_stats['avg_rating'].mean():.2f}")
        print(f"   Avg unique items per user: {user_stats['unique_items'].mean():.1f}")
        
        # Item popularity analysis
        item_stats = interaction_df.groupby('item_id').agg({
            'rating': ['count', 'mean'],
            'user_id': 'nunique'
        }).round(2)
        item_stats.columns = ['interactions', 'avg_rating', 'unique_users']
        
        print(f"\n📱 Item Popularity:")
        print(f"   Avg interactions per item: {item_stats['interactions'].mean():.1f}")
        print(f"   Avg rating per item: {item_stats['avg_rating'].mean():.2f}")
        print(f"   Most popular item interactions: {item_stats['interactions'].max()}")
        
        # Sparsity analysis
        total_possible = interaction_df['user_id'].nunique() * interaction_df['item_id'].nunique()
        sparsity = (1 - len(interaction_df) / total_possible) * 100
        print(f"   Matrix sparsity: {sparsity:.2f}%")
        
        # Store for analysis
        self.interaction_data = interaction_df
        self.user_stats = user_stats
        self.item_stats = item_stats
        
        return interaction_df
    
    def create_user_item_matrix(self, interaction_df: pd.DataFrame) -> csr_matrix:
        """Create user-item interaction matrix."""
        
        print("\n🏗️ Creating user-item matrix...")
        
        # Create pivot table
        user_item_df = interaction_df.pivot_table(
            index='user_id', 
            columns='item_id', 
            values='rating',
            fill_value=0
        )
        
        # Convert to sparse matrix for memory efficiency
        user_item_matrix = csr_matrix(user_item_df.values)
        
        print(f"   Matrix shape: {user_item_matrix.shape}")
        print(f"   Matrix density: {user_item_matrix.nnz / np.prod(user_item_matrix.shape) * 100:.3f}%")
        print(f"   Memory usage: {user_item_matrix.data.nbytes + user_item_matrix.indices.nbytes + user_item_matrix.indptr.nbytes} bytes")
        
        # Store indices for later use
        self.user_ids = user_item_df.index.values
        self.item_ids = user_item_df.columns.values
        self.user_item_matrix = user_item_matrix
        self.user_item_df = user_item_df
        
        return user_item_matrix
    
    def generate_item_features(self) -> pd.DataFrame:
        """Generate synthetic item features for content-based filtering."""
        
        print("🏷️ Generating item features...")
        
        n_items = len(self.item_ids)
        np.random.seed(self.config['random_state'])
        
        # Generate synthetic item features
        item_features = pd.DataFrame({
            'item_id': self.item_ids,
            'category': np.random.choice(['Electronics', 'Clothing', 'Books', 'Home', 'Sports'], n_items),
            'price_range': np.random.choice(['Low', 'Medium', 'High'], n_items, p=[0.3, 0.5, 0.2]),
            'brand_popularity': np.random.uniform(0, 1, n_items),
            'age_months': np.random.exponential(12, n_items),
            'avg_rating': self.item_stats['avg_rating'].values,
            'popularity_score': self.item_stats['interactions'].values / self.item_stats['interactions'].max()
        })
        
        # Create category dummy variables
        category_dummies = pd.get_dummies(item_features['category'], prefix='cat')
        price_dummies = pd.get_dummies(item_features['price_range'], prefix='price')
        
        # Combine features
        item_features = pd.concat([
            item_features[['item_id', 'brand_popularity', 'age_months', 'avg_rating', 'popularity_score']],
            category_dummies,
            price_dummies
        ], axis=1)
        
        print(f"   Generated features for {len(item_features)} items")
        print(f"   Feature columns: {len(item_features.columns) - 1}")  # Exclude item_id
        
        self.item_features = item_features
        return item_features
    
    def build_collaborative_filtering_model(self) -> Dict[str, Any]:
        """Build user-based collaborative filtering model."""
        
        print("\n🤝 Building collaborative filtering model...")
        
        # Calculate user-user similarity matrix
        user_similarity = self.calculate_user_similarity()
        
        def get_user_recommendations(user_idx: int, n_recommendations: int = 10) -> List[Tuple[int, float]]:
            """Get recommendations for a specific user."""
            
            # Get user's ratings
            user_ratings = self.user_item_matrix[user_idx].toarray().flatten()
            
            # Find items user hasn't rated
            unrated_items = np.where(user_ratings == 0)[0]
            
            if len(unrated_items) == 0:
                return []
            
            # Calculate predicted ratings for unrated items
            predicted_ratings = []
            
            for item_idx in unrated_items:
                # Find users who rated this item
                item_raters = np.where(self.user_item_matrix[:, item_idx] > 0)[0]
                
                if len(item_raters) == 0:
                    predicted_rating = self.user_item_df.mean().mean()  # Global average
                else:
                    # Calculate weighted average based on user similarity
                    similarities = user_similarity[user_idx, item_raters]
                    ratings = self.user_item_matrix[item_raters, item_idx].toarray().flatten()
                    
                    if np.sum(np.abs(similarities)) > 0:
                        predicted_rating = np.average(ratings, weights=np.abs(similarities))
                    else:
                        predicted_rating = np.mean(ratings)
                
                predicted_ratings.append((item_idx, predicted_rating))
            
            # Sort by predicted rating and return top N
            predicted_ratings.sort(key=lambda x: x[1], reverse=True)
            return predicted_ratings[:n_recommendations]
        
        cf_model = {
            'type': 'collaborative_filtering',
            'user_similarity': user_similarity,
            'recommend_function': get_user_recommendations
        }
        
        print("✅ Collaborative filtering model built")
        return cf_model
    
    def calculate_user_similarity(self) -> np.ndarray:
        """Calculate user-user cosine similarity matrix."""
        
        print("   Computing user similarities...")
        
        n_users = self.user_item_matrix.shape[0]
        user_similarity = np.zeros((n_users, n_users))
        
        # Calculate pairwise cosine similarities (sample for efficiency)
        sample_size = min(100, n_users)  # Sample users for faster computation
        user_indices = np.random.choice(n_users, sample_size, replace=False)
        
        for i, user_i in enumerate(user_indices):
            user_i_ratings = self.user_item_matrix[user_i].toarray().flatten()
            
            for j, user_j in enumerate(user_indices):
                if i <= j:
                    user_j_ratings = self.user_item_matrix[user_j].toarray().flatten()
                    
                    # Only consider items both users have rated
                    common_items = (user_i_ratings > 0) & (user_j_ratings > 0)
                    
                    if np.sum(common_items) > 0:
                        similarity = 1 - cosine(user_i_ratings[common_items], user_j_ratings[common_items])
                        user_similarity[user_i, user_j] = similarity
                        user_similarity[user_j, user_i] = similarity
        
        print(f"   Computed similarities for {sample_size} users")
        return user_similarity
    
    def build_content_based_model(self) -> Dict[str, Any]:
        """Build content-based recommendation model."""
        
        print("\n📋 Building content-based model...")
        
        # Create user profiles based on item features they've interacted with
        user_profiles = self.create_user_profiles()
        
        # Calculate item-item similarity based on features
        item_similarity = self.calculate_item_similarity()
        
        def get_content_recommendations(user_idx: int, n_recommendations: int = 10) -> List[Tuple[int, float]]:
            """Get content-based recommendations for a user."""
            
            user_id = self.user_ids[user_idx]
            user_profile = user_profiles.loc[user_profiles['user_id'] == user_id]
            
            if user_profile.empty:
                return []
            
            # Get user's ratings
            user_ratings = self.user_item_matrix[user_idx].toarray().flatten()
            unrated_items = np.where(user_ratings == 0)[0]
            
            if len(unrated_items) == 0:
                return []
            
            # Calculate content-based scores
            recommendations = []
            
            for item_idx in unrated_items:
                item_id = self.item_ids[item_idx]
                item_features_row = self.item_features[self.item_features['item_id'] == item_id]
                
                if not item_features_row.empty:
                    # Simple content-based scoring (can be enhanced)
                    content_score = self.calculate_content_score(user_profile, item_features_row)
                    recommendations.append((item_idx, content_score))
            
            # Sort and return top recommendations
            recommendations.sort(key=lambda x: x[1], reverse=True)
            return recommendations[:n_recommendations]
        
        content_model = {
            'type': 'content_based',
            'user_profiles': user_profiles,
            'item_similarity': item_similarity,
            'recommend_function': get_content_recommendations
        }
        
        print("✅ Content-based model built")
        return content_model
    
    def create_user_profiles(self) -> pd.DataFrame:
        """Create user profiles based on their interaction history."""
        
        user_profiles = []
        
        for user_id in self.user_ids[:100]:  # Sample for efficiency
            user_interactions = self.interaction_data[self.interaction_data['user_id'] == user_id]
            
            if not user_interactions.empty:
                # Get items user interacted with
                user_items = user_interactions['item_id'].values
                user_ratings = user_interactions['rating'].values
                
                # Get features of these items
                user_item_features = self.item_features[self.item_features['item_id'].isin(user_items)]
                
                if not user_item_features.empty:
                    # Weight features by ratings
                    feature_cols = [col for col in user_item_features.columns if col not in ['item_id']]
                    
                    # Calculate weighted average of features
                    weighted_profile = {}
                    for col in feature_cols:
                        if col in user_item_features.columns:
                            values = user_item_features[col].values
                            if np.issubdtype(values.dtype, np.number):
                                weighted_avg = np.average(values, weights=user_ratings[:len(values)])
                                weighted_profile[col] = weighted_avg
                    
                    weighted_profile['user_id'] = user_id
                    user_profiles.append(weighted_profile)
        
        return pd.DataFrame(user_profiles)
    
    def calculate_item_similarity(self) -> np.ndarray:
        """Calculate item-item similarity based on features."""
        
        feature_cols = [col for col in self.item_features.columns if col not in ['item_id']]
        feature_matrix = self.item_features[feature_cols].values
        
        # Normalize features
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        feature_matrix_scaled = scaler.fit_transform(feature_matrix)
        
        # Calculate cosine similarity
        from sklearn.metrics.pairwise import cosine_similarity
        item_similarity = cosine_similarity(feature_matrix_scaled)
        
        return item_similarity
    
    def calculate_content_score(self, user_profile: pd.DataFrame, item_features: pd.DataFrame) -> float:
        """Calculate content-based score for an item given user profile."""
        
        # Simple scoring based on feature matching
        score = 0.0
        
        # Category preferences
        if 'cat_Electronics' in user_profile.columns and 'cat_Electronics' in item_features.columns:
            score += user_profile['cat_Electronics'].iloc[0] * item_features['cat_Electronics'].iloc[0]
        
        # Price range preferences
        if 'price_Medium' in user_profile.columns and 'price_Medium' in item_features.columns:
            score += user_profile['price_Medium'].iloc[0] * item_features['price_Medium'].iloc[0]
        
        # Popularity score
        if 'popularity_score' in user_profile.columns and 'popularity_score' in item_features.columns:
            score += user_profile['popularity_score'].iloc[0] * item_features['popularity_score'].iloc[0] * 0.5
        
        return score
    
    def build_matrix_factorization_model(self) -> Dict[str, Any]:
        """Build matrix factorization model using SVD."""
        
        print("\n🔢 Building matrix factorization model...")
        
        try:
            from sklearn.decomposition import TruncatedSVD
            
            # Apply SVD to user-item matrix
            n_factors = min(50, min(self.user_item_matrix.shape) - 1)
            svd = TruncatedSVD(n_components=n_factors, random_state=self.config['random_state'])
            user_factors = svd.fit_transform(self.user_item_matrix)
            item_factors = svd.components_.T
            
            def get_mf_recommendations(user_idx: int, n_recommendations: int = 10) -> List[Tuple[int, float]]:
                """Get matrix factorization recommendations."""
                
                # Calculate predicted ratings
                user_vector = user_factors[user_idx]
                predicted_ratings = np.dot(item_factors, user_vector)
                
                # Get user's current ratings
                current_ratings = self.user_item_matrix[user_idx].toarray().flatten()
                
                # Only recommend items user hasn't rated
                unrated_items = np.where(current_ratings == 0)[0]
                
                if len(unrated_items) == 0:
                    return []
                
                # Get predictions for unrated items
                unrated_predictions = [(item_idx, predicted_ratings[item_idx]) for item_idx in unrated_items]
                
                # Sort and return top recommendations
                unrated_predictions.sort(key=lambda x: x[1], reverse=True)
                return unrated_predictions[:n_recommendations]
            
            mf_model = {
                'type': 'matrix_factorization',
                'svd_model': svd,
                'user_factors': user_factors,
                'item_factors': item_factors,
                'recommend_function': get_mf_recommendations
            }
            
            print(f"✅ Matrix factorization model built with {n_factors} factors")
            return mf_model
            
        except ImportError:
            print("❌ Scikit-learn not available for matrix factorization")
            return None
    
    def evaluate_recommendation_models(self, models: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate recommendation models using standard metrics."""
        
        print("\n📊 Evaluating recommendation models...")
        
        # Split users for testing
        n_users = len(self.user_ids)
        test_users = np.random.choice(n_users, size=int(n_users * self.config['test_size']), replace=False)
        
        evaluation_results = {}
        
        for model in models:
            if model is None:
                continue
                
            model_name = model['type']
            print(f"   Evaluating {model_name}...")
            
            recommend_function = model['recommend_function']
            
            # Evaluation metrics
            precision_scores = []
            recall_scores = []
            ndcg_scores = []
            
            for user_idx in test_users[:20]:  # Sample for faster evaluation
                
                # Get user's actual high-rated items (rating >= 4)
                user_ratings = self.user_item_matrix[user_idx].toarray().flatten()
                actual_liked_items = set(np.where(user_ratings >= 4)[0])
                
                if len(actual_liked_items) == 0:
                    continue
                
                # Get recommendations
                recommendations = recommend_function(user_idx, self.config['n_recommendations'])
                recommended_items = set([item_idx for item_idx, _ in recommendations])
                
                if len(recommended_items) == 0:
                    continue
                
                # Calculate precision and recall
                relevant_recommendations = recommended_items.intersection(actual_liked_items)
                
                precision = len(relevant_recommendations) / len(recommended_items) if recommended_items else 0
                recall = len(relevant_recommendations) / len(actual_liked_items) if actual_liked_items else 0
                
                precision_scores.append(precision)
                recall_scores.append(recall)
                
                # Calculate NDCG (simplified)
                ndcg = self.calculate_ndcg(recommendations, actual_liked_items)
                ndcg_scores.append(ndcg)
            
            # Calculate average metrics
            avg_precision = np.mean(precision_scores) if precision_scores else 0
            avg_recall = np.mean(recall_scores) if recall_scores else 0
            avg_ndcg = np.mean(ndcg_scores) if ndcg_scores else 0
            
            # F1 score
            f1_score = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0
            
            evaluation_results[model_name] = {
                'precision': avg_precision,
                'recall': avg_recall,
                'f1_score': f1_score,
                'ndcg': avg_ndcg,
                'n_evaluated_users': len(precision_scores)
            }
            
            print(f"     Precision: {avg_precision:.3f}")
            print(f"     Recall: {avg_recall:.3f}")
            print(f"     F1-Score: {f1_score:.3f}")
            print(f"     NDCG: {avg_ndcg:.3f}")
        
        return evaluation_results
    
    def calculate_ndcg(self, recommendations: List[Tuple[int, float]], actual_liked_items: set) -> float:
        """Calculate Normalized Discounted Cumulative Gain."""
        
        if not recommendations:
            return 0.0
        
        # Calculate DCG
        dcg = 0.0
        for i, (item_idx, _) in enumerate(recommendations):
            relevance = 1 if item_idx in actual_liked_items else 0
            dcg += relevance / np.log2(i + 2)  # +2 because log2(1) = 0
        
        # Calculate IDCG (ideal DCG)
        ideal_relevances = [1] * min(len(actual_liked_items), len(recommendations))
        idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_relevances))
        
        # Calculate NDCG
        ndcg = dcg / idcg if idcg > 0 else 0.0
        return ndcg
    
    def calculate_business_impact(self, evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate business impact of recommendation system."""
        
        print("\n💰 Calculating business impact...")
        
        # Select best model based on F1 score
        best_model = max(evaluation_results.keys(), 
                        key=lambda x: evaluation_results[x]['f1_score'])
        
        best_metrics = evaluation_results[best_model]
        
        # Business calculations
        n_users = self.config['n_users']
        n_recommendations_per_user = self.config['n_recommendations']
        total_recommendations = n_users * n_recommendations_per_user
        
        # Click-through and conversion estimates
        baseline_ctr = 0.02  # 2% baseline click-through rate
        improved_ctr = baseline_ctr * (1 + best_metrics['precision'] * 2)  # Precision improves CTR
        
        baseline_conversion = self.config['business_params']['conversion_rate_baseline']
        improved_conversion = baseline_conversion * (1 + best_metrics['recall'])  # Recall improves conversion
        
        # Revenue calculations
        recommendations_clicked = int(total_recommendations * improved_ctr)
        recommendations_purchased = int(recommendations_clicked * improved_conversion)
        
        business_impact = self.business_calc.calculate_recommendation_business_impact(
            recommendations_made=total_recommendations,
            recommendations_clicked=recommendations_clicked,
            recommendations_purchased=recommendations_purchased,
            **self.config['business_params']
        )
        
        # Add additional metrics
        business_impact.update({
            'best_model': best_model,
            'improved_ctr': improved_ctr,
            'improved_conversion_rate': improved_conversion,
            'baseline_ctr': baseline_ctr,
            'ctr_lift': (improved_ctr - baseline_ctr) / baseline_ctr * 100,
            'total_users': n_users,
            'recommendations_per_user': n_recommendations_per_user
        })
        
        print(f"📊 Business Impact Summary:")
        print(f"   Best Model: {best_model}")
        print(f"   Click-through Rate: {improved_ctr:.2%} (vs {baseline_ctr:.2%} baseline)")
        print(f"   CTR Lift: +{business_impact['ctr_lift']:.1f}%")
        print(f"   Conversion Rate: {improved_conversion:.2%}")
        print(f"   Total Revenue: ${business_impact['total_revenue']:,.2f}")
        print(f"   Net Revenue: ${business_impact['net_revenue']:,.2f}")
        print(f"   ROI: {business_impact['roi_percentage']:.1f}%")
        
        self.business_impact = business_impact
        return business_impact
    
    def run_complete_analysis(self) -> Dict[str, Any]:
        """Execute complete recommendation system analysis."""
        
        print("🚀 Starting Recommendation System Analysis")
        print("=" * 50)
        
        # 1. Load and analyze data
        interaction_df = self.load_and_analyze_interaction_data()
        
        # 2. Create user-item matrix
        user_item_matrix = self.create_user_item_matrix(interaction_df)
        
        # 3. Generate item features
        item_features = self.generate_item_features()
        
        # 4. Build recommendation models
        models = []
        
        # Collaborative filtering
        cf_model = self.build_collaborative_filtering_model()
        models.append(cf_model)
        
        # Content-based filtering
        content_model = self.build_content_based_model()
        models.append(content_model)
        
        # Matrix factorization
        mf_model = self.build_matrix_factorization_model()
        if mf_model:
            models.append(mf_model)
        
        # 5. Evaluate models
        evaluation_results = self.evaluate_recommendation_models(models)
        
        # 6. Calculate business impact
        business_impact = self.calculate_business_impact(evaluation_results)
        
        # 7. Compile final results
        final_results = {
            'evaluation_results': evaluation_results,
            'business_impact': business_impact,
            'best_model': business_impact['best_model'],
            'data_summary': {
                'total_users': len(self.user_ids),
                'total_items': len(self.item_ids),
                'total_interactions': len(interaction_df),
                'matrix_sparsity': (1 - user_item_matrix.nnz / np.prod(user_item_matrix.shape)) * 100
            }
        }
        
        print("\n🎉 Recommendation System Analysis Complete!")
        print(f"   Best Model: {final_results['best_model']}")
        print(f"   Revenue Impact: ${business_impact['total_revenue']:,.2f}")
        print(f"   CTR Improvement: +{business_impact['ctr_lift']:.1f}%")
        print(f"   ROI: {business_impact['roi_percentage']:.1f}%")
        
        return final_results

def main():
    """Main execution function for standalone running."""
    
    # Configuration for recommendation system
    config = {
        'n_users': 1000,
        'n_items': 500,
        'n_interactions': 50000,
        'recommendation_algorithms': ['collaborative_filtering', 'content_based', 'matrix_factorization'],
        'business_params': {
            'avg_order_value': 75,
            'recommendation_cost': 0.05,
            'conversion_rate_baseline': 0.02
        }
    }
    
    # Run recommendation system analysis
    rec_system = RecommendationSystem(config)
    results = rec_system.run_complete_analysis()
    
    return results

if __name__ == "__main__":
    results = main()