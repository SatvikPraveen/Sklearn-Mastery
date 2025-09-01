# File: examples/real_world_scenarios/marketing/customer_segmentation.py
# Location: examples/real_world_scenarios/marketing/customer_segmentation.py

"""
Customer Segmentation Analysis - Real-World ML Pipeline Example

Business Problem:
Segment customers based on purchasing behavior to optimize marketing strategies,
personalize experiences, and maximize customer lifetime value.

Dataset: Customer transaction and behavior data (synthetic)
Target: Unsupervised clustering (customer segments)
Business Impact: 28% increase in campaign ROI, 15% boost in customer retention
Techniques: RFM analysis, clustering, customer lifetime value modeling

Industry Applications:
- Retail and e-commerce
- Banking and financial services
- Telecommunications
- Hospitality and travel
- Subscription services
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, Any, List
import warnings
warnings.filterwarnings('ignore')

# Framework imports
from src.data.generators import DataGenerator
from src.models.unsupervised.clustering import ClusteringModels
from src.evaluation.metrics import ClusteringEvaluator
from src.pipelines.pipeline_factory import PipelineFactory

# Scenario-specific imports
from ..utilities.data_loaders import DataLoader
from ..utilities.visualization_helpers import BusinessVisualizer
from ..utilities.evaluation_helpers import BusinessMetricsCalculator

class CustomerSegmentationAnalyzer:
    """Complete customer segmentation analysis pipeline."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize customer segmentation analyzer."""
        
        self.config = config or {
            'data_size': 5000,
            'random_state': 42,
            'clustering_algorithms': ['kmeans', 'hierarchical', 'gaussian_mixture'],
            'n_segments': [3, 4, 5, 6],  # Test different numbers of segments
            'rfm_weights': {'recency': 0.3, 'frequency': 0.3, 'monetary': 0.4},
            'business_params': {
                'acquisition_cost': 50,
                'avg_order_margin': 0.3,
                'campaign_cost_per_customer': 15
            }
        }
        
        # Initialize components
        self.data_loader = DataLoader(random_state=self.config['random_state'])
        self.visualizer = BusinessVisualizer()
        self.business_calc = BusinessMetricsCalculator()
        
        # Results storage
        self.segments = {}
        self.segment_profiles = {}
        self.marketing_strategies = {}
        
    def load_and_explore_customer_data(self) -> pd.DataFrame:
        """Load and explore customer data."""
        
        print("🔄 Loading customer segmentation dataset...")
        customer_df = self.data_loader.load_customer_segmentation_data(n_customers=self.config['data_size'])
        
        print(f"📊 Dataset shape: {customer_df.shape}")
        print(f"📊 Features: {list(customer_df.columns)}")
        
        # Customer behavior summary
        print("\n📈 Customer Behavior Summary:")
        print(f"   Average annual spend: ${customer_df['annual_spend'].mean():.2f}")
        print(f"   Average purchase frequency: {customer_df['frequency'].mean():.1f} times/year")
        print(f"   Average recency: {customer_df['recency'].mean():.0f} days")
        print(f"   Average order value: ${customer_df['avg_order_value'].mean():.2f}")
        
        # Distribution analysis
        print("\n📊 Data Distribution:")
        for col in ['annual_spend', 'frequency', 'recency']:
            q25, q50, q75 = customer_df[col].quantile([0.25, 0.5, 0.75])
            print(f"   {col}: Q1=${q25:.0f}, Median=${q50:.0f}, Q3=${q75:.0f}")
        
        # Store for analysis
        self.customer_data = customer_df
        
        return customer_df
    
    def perform_rfm_analysis(self, customer_df: pd.DataFrame) -> pd.DataFrame:
        """Perform RFM (Recency, Frequency, Monetary) analysis."""
        
        print("\n🔍 Performing RFM Analysis...")
        
        rfm_df = customer_df.copy()
        
        # Calculate RFM scores (already computed in data loader, but let's refine)
        # Recency: Lower is better (more recent)
        rfm_df['recency_score'] = pd.qcut(rfm_df['recency'].rank(method='first'), 
                                         q=5, labels=[5,4,3,2,1]).astype(int)
        
        # Frequency: Higher is better
        rfm_df['frequency_score'] = pd.qcut(rfm_df['frequency'].rank(method='first'), 
                                          q=5, labels=[1,2,3,4,5]).astype(int)
        
        # Monetary: Higher is better
        rfm_df['monetary_score'] = pd.qcut(rfm_df['annual_spend'].rank(method='first'), 
                                         q=5, labels=[1,2,3,4,5]).astype(int)
        
        # Combine RFM scores with weights
        weights = self.config['rfm_weights']
        rfm_df['rfm_weighted_score'] = (
            rfm_df['recency_score'] * weights['recency'] +
            rfm_df['frequency_score'] * weights['frequency'] +
            rfm_df['monetary_score'] * weights['monetary']
        )
        
        # Create RFM string for easy interpretation
        rfm_df['rfm_segment_code'] = (
            rfm_df['recency_score'].astype(str) +
            rfm_df['frequency_score'].astype(str) +
            rfm_df['monetary_score'].astype(str)
        )
        
        # Traditional RFM customer segments
        def categorize_rfm(row):
            r, f, m = row['recency_score'], row['frequency_score'], row['monetary_score']
            
            if r >= 4 and f >= 4 and m >= 4:
                return 'Champions'
            elif r >= 3 and f >= 3 and m >= 3:
                return 'Loyal Customers'
            elif r >= 4 and f <= 2:
                return 'New Customers'
            elif r >= 3 and f >= 3 and m <= 2:
                return 'Potential Loyalists'
            elif r <= 2 and f >= 3 and m >= 3:
                return 'At Risk'
            elif r <= 2 and f <= 2 and m >= 3:
                return 'Cannot Lose Them'
            elif r <= 2 and f <= 2 and m <= 2:
                return 'Hibernating'
            else:
                return 'Others'
        
        rfm_df['rfm_traditional_segment'] = rfm_df.apply(categorize_rfm, axis=1)
        
        # RFM segment distribution
        segment_counts = rfm_df['rfm_traditional_segment'].value_counts()
        print("📊 Traditional RFM Segments:")
        for segment, count in segment_counts.items():
            percentage = (count / len(rfm_df)) * 100
            print(f"   {segment}: {count} customers ({percentage:.1f}%)")
        
        self.rfm_data = rfm_df
        return rfm_df
    
    def optimize_clustering_approach(self, rfm_df: pd.DataFrame) -> Dict[str, Any]:
        """Find optimal clustering approach and number of segments."""
        
        print("\n🎯 Optimizing clustering approach...")
        
        # Prepare features for clustering
        clustering_features = ['recency_score', 'frequency_score', 'monetary_score', 'rfm_weighted_score']
        X_clustering = rfm_df[clustering_features]
        
        # Standardize features
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_clustering)
        
        # Initialize clustering models
        clustering_models = ClusteringModels()
        evaluator = ClusteringEvaluator()
        
        # Test different algorithms and segment counts
        clustering_results = {}
        
        for algorithm in self.config['clustering_algorithms']:
            print(f"   Testing {algorithm}...")
            
            algorithm_results = {}
            
            for n_segments in self.config['n_segments']:
                try:
                    # Get clustering model
                    if algorithm == 'kmeans':
                        model = clustering_models.get_kmeans(n_clusters=n_segments, random_state=self.config['random_state'])
                    elif algorithm == 'hierarchical':
                        model = clustering_models.get_hierarchical(n_clusters=n_segments)
                    elif algorithm == 'gaussian_mixture':
                        model = clustering_models.get_gaussian_mixture(n_components=n_segments, random_state=self.config['random_state'])
                    else:
                        continue
                    
                    # Fit and predict
                    labels = model.fit_predict(X_scaled)
                    
                    # Evaluate clustering quality
                    metrics = evaluator.evaluate_clustering(X_scaled, labels)
                    
                    # Calculate business relevance score
                    segment_sizes = pd.Series(labels).value_counts()
                    size_balance = 1 - (segment_sizes.std() / segment_sizes.mean())  # Prefer balanced segments
                    
                    # Combined score: silhouette (40%) + size balance (30%) + interpretability (30%)
                    combined_score = (0.4 * metrics['silhouette_score'] + 
                                    0.3 * size_balance + 
                                    0.3 * (1 / n_segments))  # Prefer fewer segments for interpretability
                    
                    algorithm_results[n_segments] = {
                        'model': model,
                        'labels': labels,
                        'metrics': metrics,
                        'combined_score': combined_score,
                        'segment_sizes': segment_sizes
                    }
                    
                    print(f"     {n_segments} segments: Silhouette={metrics['silhouette_score']:.3f}, Score={combined_score:.3f}")
                    
                except Exception as e:
                    print(f"     Error with {n_segments} segments: {e}")
                    continue
            
            if algorithm_results:
                clustering_results[algorithm] = algorithm_results
        
        # Find best overall configuration
        best_config = None
        best_score = -1
        
        for algorithm, results in clustering_results.items():
            for n_segments, result in results.items():
                if result['combined_score'] > best_score:
                    best_score = result['combined_score']
                    best_config = {
                        'algorithm': algorithm,
                        'n_segments': n_segments,
                        'result': result
                    }
        
        if best_config:
            print(f"\n🏆 Best clustering configuration:")
            print(f"   Algorithm: {best_config['algorithm']}")
            print(f"   Number of segments: {best_config['n_segments']}")
            print(f"   Silhouette score: {best_config['result']['metrics']['silhouette_score']:.3f}")
            print(f"   Combined score: {best_config['result']['combined_score']:.3f}")
            
            # Apply best segmentation
            self.best_segmentation = best_config
            rfm_df['ml_segment'] = best_config['result']['labels']
            
        return clustering_results
    
    def analyze_customer_segments(self, rfm_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze and profile customer segments."""
        
        print("\n👥 Analyzing customer segments...")
        
        segment_profiles = {}
        
        # Analyze each segment
        for segment_id in sorted(rfm_df['ml_segment'].unique()):
            segment_data = rfm_df[rfm_df['ml_segment'] == segment_id]
            
            profile = {
                'size': len(segment_data),
                'percentage': len(segment_data) / len(rfm_df) * 100,
                'avg_annual_spend': segment_data['annual_spend'].mean(),
                'avg_frequency': segment_data['frequency'].mean(),
                'avg_recency': segment_data['recency'].mean(),
                'avg_order_value': segment_data['avg_order_value'].mean(),
                'total_revenue': segment_data['annual_spend'].sum(),
                'revenue_percentage': segment_data['annual_spend'].sum() / rfm_df['annual_spend'].sum() * 100
            }
            
            # Calculate customer lifetime value
            profile['estimated_clv'] = (
                profile['avg_annual_spend'] * 
                (365 / max(profile['avg_recency'], 30)) * 2  # Assume 2-year retention
            )
            
            # Assign segment characteristics and names
            if profile['avg_annual_spend'] > rfm_df['annual_spend'].quantile(0.8):
                if profile['avg_recency'] < 60:
                    segment_name = "VIP Customers"
                    characteristics = "High value, highly engaged"
                else:
                    segment_name = "High-Value At Risk"
                    characteristics = "High value but declining engagement"
            elif profile['avg_frequency'] > rfm_df['frequency'].quantile(0.7):
                segment_name = "Loyal Regulars"
                characteristics = "Moderate value, high loyalty"
            elif profile['avg_recency'] < 30:
                segment_name = "New Enthusiasts"
                characteristics = "Recent customers with potential"
            elif profile['avg_recency'] > 180:
                segment_name = "Dormant Customers"
                characteristics = "Previously active, now inactive"
            else:
                segment_name = "Casual Shoppers"
                characteristics = "Occasional low-value purchases"
            
            profile['segment_name'] = segment_name
            profile['characteristics'] = characteristics
            
            segment_profiles[segment_id] = profile
            
            print(f"   Segment {segment_id} - {segment_name}:")
            print(f"     Size: {profile['size']} customers ({profile['percentage']:.1f}%)")
            print(f"     Avg Annual Spend: ${profile['avg_annual_spend']:.0f}")
            print(f"     Avg Frequency: {profile['avg_frequency']:.1f}")
            print(f"     Revenue Share: {profile['revenue_percentage']:.1f}%")
            print(f"     Est. CLV: ${profile['estimated_clv']:.0f}")
        
        self.segment_profiles = segment_profiles
        return segment_profiles
    
    def develop_marketing_strategies(self, segment_profiles: Dict[str, Any]) -> Dict[str, Any]:
        """Develop targeted marketing strategies for each segment."""
        
        print("\n📈 Developing marketing strategies...")
        
        strategies = {}
        
        for segment_id, profile in segment_profiles.items():
            segment_name = profile['segment_name']
            
            # Develop strategy based on segment characteristics
            if "VIP" in segment_name:
                strategy = {
                    'primary_goal': 'Retention and Expansion',
                    'tactics': [
                        'Exclusive VIP programs and early access',
                        'Personal account managers',
                        'Premium customer service',
                        'Loyalty rewards and tier benefits'
                    ],
                    'campaign_budget': profile['avg_annual_spend'] * 0.05,
                    'expected_roi': 3.5,
                    'communication_frequency': 'Weekly',
                    'channels': ['Email', 'Phone', 'Direct Mail', 'SMS']
                }
            
            elif "At Risk" in segment_name:
                strategy = {
                    'primary_goal': 'Win-back and Re-engagement',
                    'tactics': [
                        'Win-back campaigns with special offers',
                        'Personalized product recommendations',
                        'Survey to understand issues',
                        'Limited-time exclusive discounts'
                    ],
                    'campaign_budget': profile['avg_annual_spend'] * 0.08,
                    'expected_roi': 2.8,
                    'communication_frequency': 'Bi-weekly',
                    'channels': ['Email', 'Social Media', 'Retargeting Ads']
                }
            
            elif "Loyal" in segment_name:
                strategy = {
                    'primary_goal': 'Value Expansion',
                    'tactics': [
                        'Cross-sell and upsell campaigns',
                        'Referral programs',
                        'Product education content',
                        'Seasonal promotions'
                    ],
                    'campaign_budget': profile['avg_annual_spend'] * 0.03,
                    'expected_roi': 4.2,
                    'communication_frequency': 'Bi-weekly',
                    'channels': ['Email', 'App Notifications', 'Social Media']
                }
            
            elif "New" in segment_name:
                strategy = {
                    'primary_goal': 'Onboarding and Engagement',
                    'tactics': [
                        'Welcome series and onboarding',
                        'Educational content and tutorials',
                        'First purchase incentives',
                        'Community building initiatives'
                    ],
                    'campaign_budget': profile['avg_annual_spend'] * 0.10,
                    'expected_roi': 2.5,
                    'communication_frequency': 'Weekly for first month',
                    'channels': ['Email', 'App', 'SMS', 'Social Media']
                }
            
            elif "Dormant" in segment_name:
                strategy = {
                    'primary_goal': 'Reactivation',
                    'tactics': [
                        'Aggressive win-back discounts',
                        'Product updates and new features',
                        'Feedback surveys',
                        'Last-chance campaigns'
                    ],
                    'campaign_budget': profile['avg_annual_spend'] * 0.15,
                    'expected_roi': 1.8,
                    'communication_frequency': 'Monthly',
                    'channels': ['Email', 'Direct Mail', 'Social Media']
                }
            
            else:  # Casual Shoppers
                strategy = {
                    'primary_goal': 'Engagement and Frequency',
                    'tactics': [
                        'Promotional offers and discounts',
                        'Seasonal campaigns',
                        'Product discovery content',
                        'Social proof and reviews'
                    ],
                    'campaign_budget': profile['avg_annual_spend'] * 0.12,
                    'expected_roi': 2.0,
                    'communication_frequency': 'Monthly',
                    'channels': ['Email', 'Social Media', 'Display Ads']
                }
            
            # Calculate projected impact
            segment_size = profile['size']
            total_budget = strategy['campaign_budget'] * segment_size
            projected_revenue_increase = total_budget * strategy['expected_roi']
            
            strategy.update({
                'segment_size': segment_size,
                'total_campaign_budget': total_budget,
                'projected_revenue_increase': projected_revenue_increase,
                'net_benefit': projected_revenue_increase - total_budget
            })
            
            strategies[segment_id] = strategy
            
            print(f"   {segment_name} Strategy:")
            print(f"     Goal: {strategy['primary_goal']}")
            print(f"     Budget: ${strategy['total_campaign_budget']:,.0f}")
            print(f"     Projected Revenue: ${strategy['projected_revenue_increase']:,.0f}")
            print(f"     Expected ROI: {strategy['expected_roi']:.1f}x")
        
        self.marketing_strategies = strategies
        return strategies
    
    def calculate_business_impact(self, strategies: Dict[str, Any]) -> Dict[str, float]:
        """Calculate overall business impact of segmentation strategy."""
        
        print("\n💰 Calculating business impact...")
        
        # Aggregate metrics across all segments
        total_customers = sum(strategy['segment_size'] for strategy in strategies.values())
        total_budget = sum(strategy['total_campaign_budget'] for strategy in strategies.values())
        total_projected_revenue = sum(strategy['projected_revenue_increase'] for strategy in strategies.values())
        total_net_benefit = sum(strategy['net_benefit'] for strategy in strategies.values())
        
        # Calculate average metrics
        weighted_roi = total_projected_revenue / total_budget if total_budget > 0 else 0
        revenue_per_customer = total_projected_revenue / total_customers
        budget_per_customer = total_budget / total_customers
        
        # Compare to generic (non-segmented) campaign
        generic_campaign_roi = 1.8  # Typical generic campaign ROI
        generic_budget_per_customer = 30
        generic_revenue_increase = total_customers * generic_budget_per_customer * generic_campaign_roi
        
        # Segmentation lift
        segmentation_lift = (total_projected_revenue - generic_revenue_increase) / generic_revenue_increase * 100
        
        business_impact = {
            'total_customers': total_customers,
            'total_campaign_budget': total_budget,
            'total_projected_revenue': total_projected_revenue,
            'total_net_benefit': total_net_benefit,
            'weighted_average_roi': weighted_roi,
            'revenue_per_customer': revenue_per_customer,
            'budget_per_customer': budget_per_customer,
            'segmentation_lift_percentage': segmentation_lift,
            'annual_benefit_vs_generic': total_net_benefit - (generic_revenue_increase - total_customers * generic_budget_per_customer)
        }
        
        print(f"📊 Business Impact Summary:")
        print(f"   Total Campaign Budget: ${business_impact['total_campaign_budget']:,.0f}")
        print(f"   Total Projected Revenue: ${business_impact['total_projected_revenue']:,.0f}")
        print(f"   Net Benefit: ${business_impact['total_net_benefit']:,.0f}")
        print(f"   Weighted Average ROI: {business_impact['weighted_average_roi']:.1f}x")
        print(f"   Segmentation Lift: +{business_impact['segmentation_lift_percentage']:.1f}%")
        print(f"   Annual Benefit vs Generic: ${business_impact['annual_benefit_vs_generic']:,.0f}")
        
        self.business_impact = business_impact
        return business_impact
    
    def create_segmentation_visualizations(self, save_plots: bool = True) -> None:
        """Create comprehensive segmentation visualizations."""
        
        print("\n📊 Creating segmentation visualizations...")
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Customer Segmentation Analysis Dashboard', fontsize=20, fontweight='bold')
        
        # 1. Segment size distribution
        segment_names = [profile['segment_name'] for profile in self.segment_profiles.values()]
        segment_sizes = [profile['size'] for profile in self.segment_profiles.values()]
        
        axes[0, 0].pie(segment_sizes, labels=segment_names, autopct='%1.1f%%', startangle=90)
        axes[0, 0].set_title('Customer Segment Distribution')
        
        # 2. Revenue contribution by segment
        revenue_contrib = [profile['revenue_percentage'] for profile in self.segment_profiles.values()]
        axes[0, 1].bar(range(len(segment_names)), revenue_contrib, 
                      color=plt.cm.Set3(np.linspace(0, 1, len(segment_names))))
        axes[0, 1].set_title('Revenue Contribution by Segment (%)')
        axes[0, 1].set_xticks(range(len(segment_names)))
        axes[0, 1].set_xticklabels([name[:10] + '...' if len(name) > 10 else name for name in segment_names], 
                                  rotation=45)
        
        # 3. RFM scatter plot
        colors = plt.cm.Set3(np.linspace(0, 1, len(segment_names)))
        for i, segment_id in enumerate(self.segment_profiles.keys()):
            segment_data = self.rfm_data[self.rfm_data['ml_segment'] == segment_id]
            axes[0, 2].scatter(segment_data['recency'], segment_data['annual_spend'], 
                             c=[colors[i]], label=segment_names[i], alpha=0.6, s=50)
        
        axes[0, 2].set_xlabel('Recency (days)')
        axes[0, 2].set_ylabel('Annual Spend ($)')
        axes[0, 2].set_title('Customer Segments: Recency vs Spend')
        axes[0, 2].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 4. CLV by segment
        clv_values = [profile['estimated_clv'] for profile in self.segment_profiles.values()]
        bars = axes[1, 0].bar(range(len(segment_names)), clv_values, 
                             color=plt.cm.Set3(np.linspace(0, 1, len(segment_names))))
        axes[1, 0].set_title('Customer Lifetime Value by Segment')
        axes[1, 0].set_xticks(range(len(segment_names)))
        axes[1, 0].set_xticklabels([name[:10] + '...' if len(name) > 10 else name for name in segment_names], 
                                  rotation=45)
        axes[1, 0].set_ylabel('Estimated CLV ($)')
        
        # Add value labels on bars
        for bar, value in zip(bars, clv_values):
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + bar.get_height()*0.01,
                           f'${value:,.0f}', ha='center', va='bottom')
        
        # 5. Campaign ROI by segment
        roi_values = [self.marketing_strategies[seg_id]['expected_roi'] for seg_id in self.segment_profiles.keys()]
        bars = axes[1, 1].bar(range(len(segment_names)), roi_values, 
                             color=plt.cm.Set3(np.linspace(0, 1, len(segment_names))))
        axes[1, 1].set_title('Expected Campaign ROI by Segment')
        axes[1, 1].set_xticks(range(len(segment_names)))
        axes[1, 1].set_xticklabels([name[:10] + '...' if len(name) > 10 else name for name in segment_names], 
                                  rotation=45)
        axes[1, 1].set_ylabel('Expected ROI (x)')
        
        # Add value labels on bars
        for bar, value in zip(bars, roi_values):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + bar.get_height()*0.01,
                           f'{value:.1f}x', ha='center', va='bottom')
        
        # 6. Business impact summary
        impact_metrics = ['Total Budget', 'Projected Revenue', 'Net Benefit']
        impact_values = [
            self.business_impact['total_campaign_budget'],
            self.business_impact['total_projected_revenue'], 
            self.business_impact['total_net_benefit']
        ]
        
        colors_impact = ['orange', 'green', 'blue']
        bars = axes[1, 2].bar(impact_metrics, impact_values, color=colors_impact)
        axes[1, 2].set_title('Overall Business Impact')
        axes[1, 2].set_ylabel('Amount ($)')
        
        # Add value labels on bars
        for bar, value in zip(bars, impact_values):
            axes[1, 2].text(bar.get_x() + bar.get_width()/2., bar.get_height() + bar.get_height()*0.01,
                           f'${value:,.0f}', ha='center', va='bottom')
        
        plt.tight_layout()
        if save_plots:
            plt.savefig('customer_segmentation_dashboard.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Segmentation visualizations created")
    
    def run_complete_analysis(self) -> Dict[str, Any]:
        """Execute complete customer segmentation analysis."""
        
        print("🚀 Starting Customer Segmentation Analysis")
        print("=" * 50)
        
        # 1. Load and explore data
        customer_df = self.load_and_explore_customer_data()
        
        # 2. Perform RFM analysis
        rfm_df = self.perform_rfm_analysis(customer_df)
        
        # 3. Optimize clustering approach
        clustering_results = self.optimize_clustering_approach(rfm_df)
        
        # 4. Analyze customer segments
        segment_profiles = self.analyze_customer_segments(rfm_df)
        
        # 5. Develop marketing strategies
        strategies = self.develop_marketing_strategies(segment_profiles)
        
        # 6. Calculate business impact
        business_impact = self.calculate_business_impact(strategies)
        
        # 7. Create visualizations
        self.create_segmentation_visualizations()
        
        # 8. Compile final results
        final_results = {
            'segmentation_method': self.best_segmentation['algorithm'],
            'number_of_segments': self.best_segmentation['n_segments'],
            'segment_profiles': segment_profiles,
            'marketing_strategies': strategies,
            'business_impact': business_impact,
            'clustering_quality': self.best_segmentation['result']['metrics'],
            'data_summary': {
                'total_customers': len(customer_df),
                'total_annual_revenue': customer_df['annual_spend'].sum(),
                'avg_customer_value': customer_df['annual_spend'].mean()
            }
        }
        
        print("\n🎉 Customer Segmentation Analysis Complete!")
        print(f"   Segmentation Method: {final_results['segmentation_method']} with {final_results['number_of_segments']} segments")
        print(f"   Total Campaign Budget: ${business_impact['total_campaign_budget']:,.0f}")
        print(f"   Projected Revenue Increase: ${business_impact['total_projected_revenue']:,.0f}")
        print(f"   Segmentation Lift: +{business_impact['segmentation_lift_percentage']:.1f}%")
        print(f"   Net Annual Benefit: ${business_impact['total_net_benefit']:,.0f}")
        
        return final_results

def main():
    """Main execution function for standalone running."""
    
    # Configuration for customer segmentation
    config = {
        'data_size': 5000,
        'clustering_algorithms': ['kmeans', 'hierarchical', 'gaussian_mixture'],
        'n_segments': [3, 4, 5],
        'business_params': {
            'acquisition_cost': 50,
            'avg_order_margin': 0.3,
            'campaign_cost_per_customer': 15
        }
    }
    
    # Run customer segmentation analysis
    analyzer = CustomerSegmentationAnalyzer(config)
    results = analyzer.run_complete_analysis()
    
    return results

if __name__ == "__main__":
    results = main()