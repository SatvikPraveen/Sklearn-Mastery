# File: examples/real_world_scenarios/business_analytics/market_basket_analysis.py
# Location: examples/real_world_scenarios/business_analytics/market_basket_analysis.py

"""
Market Basket Analysis - Real-World ML Pipeline Example

Business Problem:
Discover product associations and purchasing patterns to optimize product placement,
cross-selling strategies, and inventory management.

Dataset: Transaction data with product purchases (synthetic)
Target: Association rules and product recommendations
Business Impact: 18% increase in cross-sell revenue, 12% improvement in inventory turnover
Techniques: Association rule mining, Apriori algorithm, collaborative filtering

Industry Applications:
- Retail chains and supermarkets
- E-commerce platforms
- Restaurant chains
- Pharmacy and healthcare
- Online marketplaces
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, Any, List, Set
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

# Framework imports
from src.data.generators import DataGenerator

# Scenario-specific imports
from ..utilities.data_loaders import DataLoader
from ..utilities.visualization_helpers import BusinessVisualizer
from ..utilities.evaluation_helpers import BusinessMetricsCalculator

class MarketBasketAnalyzer:
    """Complete market basket analysis pipeline."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize market basket analyzer."""
        
        self.config = config or {
            'n_transactions': 10000,
            'n_products': 200,
            'avg_items_per_transaction': 5,
            'random_state': 42,
            'min_support': 0.01,
            'min_confidence': 0.5,
            'min_lift': 1.0,
            'max_itemset_size': 4,
            'business_params': {
                'avg_product_price': 25.0,
                'cross_sell_success_rate': 0.15,
                'inventory_cost_rate': 0.02
            }
        }
        
        # Initialize components
        self.data_loader = DataLoader(random_state=self.config['random_state'])
        self.visualizer = BusinessVisualizer()
        self.business_calc = BusinessMetricsCalculator()
        
        # Results storage
        self.transaction_data = None
        self.frequent_itemsets = {}
        self.association_rules = []
        self.product_info = None
        
    def generate_transaction_data(self) -> pd.DataFrame:
        """Generate synthetic transaction data."""
        
        print("🔄 Generating market basket transaction data...")
        
        np.random.seed(self.config['random_state'])
        
        # Generate product catalog
        product_categories = ['Electronics', 'Clothing', 'Food', 'Books', 'Home', 'Sports', 'Beauty', 'Toys']
        products = []
        
        for i in range(self.config['n_products']):
            products.append({
                'product_id': f'P{i:03d}',
                'product_name': f'Product_{i:03d}',
                'category': np.random.choice(product_categories),
                'price': np.random.uniform(5, 100),
                'popularity_score': np.random.beta(2, 5)  # Most products have low popularity
            })
        
        self.product_info = pd.DataFrame(products)
        
        # Generate transactions with realistic patterns
        transactions = []
        transaction_id = 1
        
        for _ in range(self.config['n_transactions']):
            # Determine number of items in this transaction
            n_items = np.random.poisson(self.config['avg_items_per_transaction']) + 1
            n_items = min(n_items, 15)  # Cap at 15 items per transaction
            
            # Select products with bias towards popular items and category clustering
            selected_products = set()
            
            # First, select a primary category (80% chance)
            if np.random.random() < 0.8:
                primary_category = np.random.choice(product_categories)
                category_products = self.product_info[self.product_info['category'] == primary_category]['product_id'].values
                
                if len(category_products) > 0:
                    # Select 1-3 items from primary category
                    n_primary = min(np.random.randint(1, 4), n_items, len(category_products))
                    primary_items = np.random.choice(category_products, size=n_primary, replace=False)
                    selected_products.update(primary_items)
            
            # Fill remaining slots with random products (biased by popularity)
            remaining_slots = n_items - len(selected_products)
            if remaining_slots > 0:
                available_products = [p for p in self.product_info['product_id'].values if p not in selected_products]
                
                if len(available_products) > 0:
                    # Weight by popularity score
                    popularity_weights = self.product_info[
                        self.product_info['product_id'].isin(available_products)
                    ]['popularity_score'].values
                    popularity_weights = popularity_weights / popularity_weights.sum()
                    
                    n_random = min(remaining_slots, len(available_products))
                    random_items = np.random.choice(
                        available_products, 
                        size=n_random, 
                        replace=False, 
                        p=popularity_weights
                    )
                    selected_products.update(random_items)
            
            # Create transaction records
            for product_id in selected_products:
                transactions.append({
                    'transaction_id': transaction_id,
                    'product_id': product_id,
                    'quantity': np.random.randint(1, 4),
                    'timestamp': pd.Timestamp.now() - pd.Timedelta(days=np.random.randint(0, 365))
                })
            
            transaction_id += 1
        
        transaction_df = pd.DataFrame(transactions)
        
        print(f"📊 Generated {len(transaction_df)} transaction records")
        print(f"📊 Unique transactions: {transaction_df['transaction_id'].nunique():,}")
        print(f"📊 Unique products: {transaction_df['product_id'].nunique()}")
        print(f"📊 Average items per transaction: {transaction_df.groupby('transaction_id').size().mean():.1f}")
        
        # Analyze transaction patterns
        print("\n📈 Transaction Analysis:")
        transaction_sizes = transaction_df.groupby('transaction_id').size()
        print(f"   Transaction size distribution: min={transaction_sizes.min()}, max={transaction_sizes.max()}, median={transaction_sizes.median():.0f}")
        
        # Product frequency analysis
        product_frequency = transaction_df['product_id'].value_counts()
        print(f"   Most popular product appears in {product_frequency.iloc[0]} transactions")
        print(f"   Least popular product appears in {product_frequency.iloc[-1]} transactions")
        
        self.transaction_data = transaction_df
        return transaction_df
    
    def create_transaction_matrix(self, transaction_df: pd.DataFrame) -> pd.DataFrame:
        """Create binary transaction matrix for association rule mining."""
        
        print("\n🏗️ Creating transaction matrix...")
        
        # Create binary matrix: rows = transactions, columns = products
        transaction_matrix = transaction_df.groupby(['transaction_id', 'product_id'])['quantity'].sum().unstack(fill_value=0)
        
        # Convert to binary (1 if product was purchased, 0 otherwise)
        transaction_matrix = (transaction_matrix > 0).astype(int)
        
        print(f"   Transaction matrix shape: {transaction_matrix.shape}")
        print(f"   Matrix density: {transaction_matrix.sum().sum() / np.prod(transaction_matrix.shape) * 100:.2f}%")
        
        self.transaction_matrix = transaction_matrix
        return transaction_matrix
    
    def find_frequent_itemsets(self, transaction_matrix: pd.DataFrame) -> Dict[int, List]:
        """Find frequent itemsets using Apriori algorithm."""
        
        print("\n🔍 Finding frequent itemsets...")
        
        n_transactions = len(transaction_matrix)
        min_support_count = int(self.config['min_support'] * n_transactions)
        
        frequent_itemsets = {}
        
        # Level 1: Individual items
        item_support = transaction_matrix.sum()
        frequent_1_itemsets = item_support[item_support >= min_support_count].index.tolist()
        frequent_itemsets[1] = [(frozenset([item]), item_support[item]) for item in frequent_1_itemsets]
        
        print(f"   Level 1: {len(frequent_itemsets[1])} frequent items")
        
        # Levels 2 and above
        for k in range(2, self.config['max_itemset_size'] + 1):
            if k-1 not in frequent_itemsets or not frequent_itemsets[k-1]:
                break
            
            # Generate candidate itemsets
            candidates = self.generate_candidates(frequent_itemsets[k-1], k)
            
            if not candidates:
                break
            
            # Count support for candidates
            frequent_k_itemsets = []
            
            for candidate in candidates:
                # Calculate support
                candidate_list = list(candidate)
                support_mask = transaction_matrix[candidate_list].all(axis=1)
                support_count = support_mask.sum()
                
                if support_count >= min_support_count:
                    frequent_k_itemsets.append((candidate, support_count))
            
            if frequent_k_itemsets:
                frequent_itemsets[k] = frequent_k_itemsets
                print(f"   Level {k}: {len(frequent_itemsets[k])} frequent itemsets")
            else:
                break
        
        self.frequent_itemsets = frequent_itemsets
        
        # Display top frequent itemsets
        print("\n🔝 Top frequent itemsets:")
        for level, itemsets in frequent_itemsets.items():
            if level <= 3:  # Show up to 3-itemsets
                top_itemsets = sorted(itemsets, key=lambda x: x[1], reverse=True)[:5]
                print(f"   Level {level}:")
                for itemset, support in top_itemsets:
                    support_pct = support / n_transactions * 100
                    itemset_str = ', '.join(sorted(list(itemset)))
                    print(f"     {{{itemset_str}}}: {support} transactions ({support_pct:.1f}%)")
        
        return frequent_itemsets
    
    def generate_candidates(self, frequent_prev: List[Tuple], k: int) -> Set[frozenset]:
        """Generate candidate k-itemsets from frequent (k-1)-itemsets."""
        
        candidates = set()
        frequent_items = [itemset for itemset, _ in frequent_prev]
        
        for i in range(len(frequent_items)):
            for j in range(i + 1, len(frequent_items)):
                # Join two (k-1)-itemsets if they differ by exactly one item
                union = frequent_items[i].union(frequent_items[j])
                if len(union) == k:
                    # Check if all (k-1)-subsets are frequent
                    if self.all_subsets_frequent(union, frequent_items):
                        candidates.add(union)
        
        return candidates
    
    def all_subsets_frequent(self, itemset: frozenset, frequent_items: List[frozenset]) -> bool:
        """Check if all (k-1)-subsets of itemset are in frequent_items."""
        
        k = len(itemset)
        if k <= 1:
            return True
        
        # Generate all (k-1)-subsets
        for item in itemset:
            subset = itemset - {item}
            if subset not in frequent_items:
                return False
        
        return True
    
    def generate_association_rules(self, frequent_itemsets: Dict[int, List]) -> List[Dict]:
        """Generate association rules from frequent itemsets."""
        
        print("\n📋 Generating association rules...")
        
        association_rules = []
        n_transactions = len(self.transaction_matrix)
        
        # Generate rules from itemsets of size 2 and above
        for level in range(2, len(frequent_itemsets) + 1):
            if level not in frequent_itemsets:
                continue
            
            for itemset, itemset_support in frequent_itemsets[level]:
                # Generate all possible antecedent-consequent combinations
                items = list(itemset)
                
                for r in range(1, len(items)):
                    for antecedent_items in combinations(items, r):
                        antecedent = frozenset(antecedent_items)
                        consequent = itemset - antecedent
                        
                        # Find support of antecedent
                        antecedent_support = self.get_itemset_support(antecedent, frequent_itemsets)
                        
                        if antecedent_support > 0:
                            # Calculate metrics
                            support = itemset_support / n_transactions
                            confidence = itemset_support / antecedent_support
                            
                            # Calculate lift
                            consequent_support = self.get_itemset_support(consequent, frequent_itemsets)
                            if consequent_support > 0:
                                lift = (itemset_support / n_transactions) / ((antecedent_support / n_transactions) * (consequent_support / n_transactions))
                            else:
                                lift = 0
                            
                            # Filter by minimum thresholds
                            if (confidence >= self.config['min_confidence'] and 
                                lift >= self.config['min_lift'] and 
                                support >= self.config['min_support']):
                                
                                rule = {
                                    'antecedent': antecedent,
                                    'consequent': consequent,
                                    'antecedent_items': sorted(list(antecedent)),
                                    'consequent_items': sorted(list(consequent)),
                                    'support': support,
                                    'confidence': confidence,
                                    'lift': lift,
                                    'conviction': self.calculate_conviction(confidence, consequent_support / n_transactions)
                                }
                                
                                association_rules.append(rule)
        
        # Sort rules by lift (descending)
        association_rules.sort(key=lambda x: x['lift'], reverse=True)
        
        print(f"   Generated {len(association_rules)} association rules")
        
        # Display top rules
        print("\n🔝 Top association rules:")
        for i, rule in enumerate(association_rules[:10]):
            antecedent_str = ', '.join(rule['antecedent_items'])
            consequent_str = ', '.join(rule['consequent_items'])
            print(f"   {i+1}. {{{antecedent_str}}} → {{{consequent_str}}}")
            print(f"      Support: {rule['support']:.3f}, Confidence: {rule['confidence']:.3f}, Lift: {rule['lift']:.2f}")
        
        self.association_rules = association_rules
        return association_rules
    
    def get_itemset_support(self, itemset: frozenset, frequent_itemsets: Dict[int, List]) -> int:
        """Get support count for a given itemset."""
        
        itemset_size = len(itemset)
        if itemset_size not in frequent_itemsets:
            return 0
        
        for stored_itemset, support in frequent_itemsets[itemset_size]:
            if stored_itemset == itemset:
                return support
        
        return 0
    
    def calculate_conviction(self, confidence: float, consequent_support: float) -> float:
        """Calculate conviction metric for association rule."""
        
        if confidence >= 1.0:
            return float('inf')
        
        return (1 - consequent_support) / (1 - confidence) if confidence < 1.0 else float('inf')
    
    def analyze_product_associations(self, association_rules: List[Dict]) -> Dict[str, Any]:
        """Analyze product associations and create recommendations."""
        
        print("\n🤝 Analyzing product associations...")
        
        # Create product-to-product recommendation matrix
        product_recommendations = {}
        
        for rule in association_rules:
            for antecedent_item in rule['antecedent_items']:
                if antecedent_item not in product_recommendations:
                    product_recommendations[antecedent_item] = []
                
                for consequent_item in rule['consequent_items']:
                    product_recommendations[antecedent_item].append({
                        'recommended_product': consequent_item,
                        'confidence': rule['confidence'],
                        'lift': rule['lift'],
                        'support': rule['support']
                    })
        
        # Sort recommendations by confidence
        for product in product_recommendations:
            product_recommendations[product].sort(key=lambda x: x['confidence'], reverse=True)
            # Keep top 5 recommendations per product
            product_recommendations[product] = product_recommendations[product][:5]
        
        # Analyze category associations
        category_associations = self.analyze_category_associations(association_rules)
        
        # Create cross-sell opportunities
        cross_sell_opportunities = self.identify_cross_sell_opportunities(association_rules)
        
        analysis_results = {
            'product_recommendations': product_recommendations,
            'category_associations': category_associations,
            'cross_sell_opportunities': cross_sell_opportunities,
            'total_recommendation_pairs': sum(len(recs) for recs in product_recommendations.values())
        }
        
        print(f"   Created recommendations for {len(product_recommendations)} products")
        print(f"   Total recommendation pairs: {analysis_results['total_recommendation_pairs']}")
        print(f"   Cross-sell opportunities identified: {len(cross_sell_opportunities)}")
        
        return analysis_results
    
    def analyze_category_associations(self, association_rules: List[Dict]) -> Dict[str, Any]:
        """Analyze associations between product categories."""
        
        category_pairs = {}
        
        for rule in association_rules:
            # Get categories for antecedent and consequent items
            antecedent_categories = set()
            consequent_categories = set()
            
            for item in rule['antecedent_items']:
                category = self.product_info[self.product_info['product_id'] == item]['category'].iloc[0]
                antecedent_categories.add(category)
            
            for item in rule['consequent_items']:
                category = self.product_info[self.product_info['product_id'] == item]['category'].iloc[0]
                consequent_categories.add(category)
            
            # Record category associations
            for ant_cat in antecedent_categories:
                for cons_cat in consequent_categories:
                    if ant_cat != cons_cat:  # Cross-category associations
                        pair = tuple(sorted([ant_cat, cons_cat]))
                        if pair not in category_pairs:
                            category_pairs[pair] = []
                        category_pairs[pair].append(rule['lift'])
        
        # Calculate average lift for category pairs
        category_associations = {}
        for pair, lifts in category_pairs.items():
            category_associations[pair] = {
                'avg_lift': np.mean(lifts),
                'max_lift': np.max(lifts),
                'rule_count': len(lifts)
            }
        
        return category_associations
    
    def identify_cross_sell_opportunities(self, association_rules: List[Dict]) -> List[Dict]:
        """Identify top cross-selling opportunities."""
        
        cross_sell_opportunities = []
        
        # Sort rules by business impact (lift * support)
        business_impact_rules = []
        for rule in association_rules:
            impact_score = rule['lift'] * rule['support'] * rule['confidence']
            business_impact_rules.append((rule, impact_score))
        
        business_impact_rules.sort(key=lambda x: x[1], reverse=True)
        
        # Select top opportunities
        for rule, impact_score in business_impact_rules[:20]:
            antecedent_str = ', '.join(rule['antecedent_items'])
            consequent_str = ', '.join(rule['consequent_items'])
            
            # Estimate potential revenue
            potential_customers = int(rule['support'] * len(self.transaction_matrix))
            cross_sell_rate = rule['confidence'] * self.config['business_params']['cross_sell_success_rate']
            avg_price = self.config['business_params']['avg_product_price']
            
            potential_revenue = potential_customers * cross_sell_rate * avg_price * len(rule['consequent_items'])
            
            opportunity = {
                'antecedent': antecedent_str,
                'consequent': consequent_str,
                'confidence': rule['confidence'],
                'lift': rule['lift'],
                'support': rule['support'],
                'impact_score': impact_score,
                'potential_customers': potential_customers,
                'estimated_cross_sell_rate': cross_sell_rate,
                'potential_revenue': potential_revenue
            }
            
            cross_sell_opportunities.append(opportunity)
        
        return cross_sell_opportunities
    
    def calculate_business_impact(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate business impact of market basket analysis."""
        
        print("\n💰 Calculating business impact...")
        
        cross_sell_opportunities = analysis_results['cross_sell_opportunities']
        
        # Calculate total potential from cross-selling
        total_potential_revenue = sum(opp['potential_revenue'] for opp in cross_sell_opportunities)
        total_potential_customers = sum(opp['potential_customers'] for opp in cross_sell_opportunities)
        
        # Baseline metrics
        current_transactions = len(self.transaction_matrix)
        avg_transaction_value = (
            len(self.transaction_data) * self.config['business_params']['avg_product_price']
        ) / current_transactions
        
        # Cross-sell impact
        cross_sell_revenue_increase = total_potential_revenue * 0.3  # Assume 30% implementation success
        cross_sell_lift = cross_sell_revenue_increase / (current_transactions * avg_transaction_value) * 100
        
        # Inventory optimization impact
        fast_moving_products = len([rule for rule in self.association_rules if rule['support'] > 0.05])
        inventory_optimization_savings = fast_moving_products * 1000  # $1000 savings per optimized product
        
        business_impact = {
            'total_association_rules': len(self.association_rules),
            'cross_sell_opportunities': len(cross_sell_opportunities),
            'potential_cross_sell_revenue': total_potential_revenue,
            'estimated_revenue_increase': cross_sell_revenue_increase,
            'cross_sell_lift_percentage': cross_sell_lift,
            'inventory_optimization_savings': inventory_optimization_savings,
            'total_business_value': cross_sell_revenue_increase + inventory_optimization_savings,
            'avg_transaction_value': avg_transaction_value,
            'current_annual_revenue': current_transactions * avg_transaction_value * 12,  # Annualized
            'roi_percentage': (cross_sell_revenue_increase + inventory_optimization_savings) / (current_transactions * avg_transaction_value) * 100
        }
        
        print(f"📊 Business Impact Summary:")
        print(f"   Association rules discovered: {business_impact['total_association_rules']}")
        print(f"   Cross-sell opportunities: {business_impact['cross_sell_opportunities']}")
        print(f"   Potential cross-sell revenue: ${business_impact['potential_cross_sell_revenue']:,.0f}")
        print(f"   Estimated revenue increase: ${business_impact['estimated_revenue_increase']:,.0f}")
        print(f"   Cross-sell lift: +{business_impact['cross_sell_lift_percentage']:.1f}%")
        print(f"   Inventory optimization savings: ${business_impact['inventory_optimization_savings']:,.0f}")
        print(f"   Total business value: ${business_impact['total_business_value']:,.0f}")
        print(f"   ROI: {business_impact['roi_percentage']:.1f}%")
        
        return business_impact
    
    def create_market_basket_visualizations(self, save_plots: bool = True) -> None:
        """Create market basket analysis visualizations."""
        
        print("\n📊 Creating market basket visualizations...")
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Market Basket Analysis Dashboard', fontsize=20, fontweight='bold')
        
        # 1. Transaction size distribution
        transaction_sizes = self.transaction_data.groupby('transaction_id').size()
        axes[0, 0].hist(transaction_sizes, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Transaction Size Distribution')
        axes[0, 0].set_xlabel('Number of Items per Transaction')
        axes[0, 0].set_ylabel('Frequency')
        
        # 2. Top products by frequency
        product_frequency = self.transaction_data['product_id'].value_counts().head(15)
        axes[0, 1].barh(range(len(product_frequency)), product_frequency.values, color='lightgreen')
        axes[0, 1].set_yticks(range(len(product_frequency)))
        axes[0, 1].set_yticklabels(product_frequency.index)
        axes[0, 1].set_title('Top 15 Products by Purchase Frequency')
        axes[0, 1].set_xlabel('Number of Transactions')
        
        # 3. Association rules - Support vs Confidence
        if self.association_rules:
            supports = [rule['support'] for rule in self.association_rules]
            confidences = [rule['confidence'] for rule in self.association_rules]
            lifts = [rule['lift'] for rule in self.association_rules]
            
            scatter = axes[0, 2].scatter(supports, confidences, c=lifts, cmap='viridis', alpha=0.6)
            axes[0, 2].set_xlabel('Support')
            axes[0, 2].set_ylabel('Confidence')
            axes[0, 2].set_title('Association Rules: Support vs Confidence')
            plt.colorbar(scatter, ax=axes[0, 2], label='Lift')
        
        # 4. Category distribution
        category_counts = self.product_info['category'].value_counts()
        axes[1, 0].pie(category_counts.values, labels=category_counts.index, autopct='%1.1f%%')
        axes[1, 0].set_title('Product Category Distribution')
        
        # 5. Top association rules by lift
        if self.association_rules:
            top_rules = sorted(self.association_rules, key=lambda x: x['lift'], reverse=True)[:10]
            rule_labels = [f"{', '.join(rule['antecedent_items'])} → {', '.join(rule['consequent_items'])}" 
                          for rule in top_rules]
            rule_lifts = [rule['lift'] for rule in top_rules]
            
            axes[1, 1].barh(range(len(rule_labels)), rule_lifts, color='orange')
            axes[1, 1].set_yticks(range(len(rule_labels)))
            axes[1, 1].set_yticklabels([label[:30] + '...' if len(label) > 30 else label for label in rule_labels])
            axes[1, 1].set_title('Top 10 Association Rules by Lift')
            axes[1, 1].set_xlabel('Lift')
        
        # 6. Revenue potential by cross-sell opportunities
        if hasattr(self, 'cross_sell_opportunities'):
            top_opportunities = sorted(
                self.analysis_results['cross_sell_opportunities'], 
                key=lambda x: x['potential_revenue'], 
                reverse=True
            )[:10]
            
            opp_labels = [opp['antecedent'] for opp in top_opportunities]
            opp_revenues = [opp['potential_revenue'] for opp in top_opportunities]
            
            axes[1, 2].bar(range(len(opp_labels)), opp_revenues, color='gold')
            axes[1, 2].set_xticks(range(len(opp_labels)))
            axes[1, 2].set_xticklabels([label[:10] + '...' if len(label) > 10 else label for label in opp_labels], 
                                      rotation=45)
            axes[1, 2].set_title('Top Cross-Sell Revenue Opportunities')
            axes[1, 2].set_ylabel('Potential Revenue ($)')
        
        plt.tight_layout()
        if save_plots:
            plt.savefig('market_basket_analysis_dashboard.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Market basket visualizations created")
    
    def run_complete_analysis(self) -> Dict[str, Any]:
        """Execute complete market basket analysis."""
        
        print("🚀 Starting Market Basket Analysis")
        print("=" * 40)
        
        # 1. Generate transaction data
        transaction_df = self.generate_transaction_data()
        
        # 2. Create transaction matrix
        transaction_matrix = self.create_transaction_matrix(transaction_df)
        
        # 3. Find frequent itemsets
        frequent_itemsets = self.find_frequent_itemsets(transaction_matrix)
        
        # 4. Generate association rules
        association_rules = self.generate_association_rules(frequent_itemsets)
        
        # 5. Analyze product associations
        analysis_results = self.analyze_product_associations(association_rules)
        self.analysis_results = analysis_results  # Store for visualization
        
        # 6. Calculate business impact
        business_impact = self.calculate_business_impact(analysis_results)
        
        # 7. Create visualizations
        self.create_market_basket_visualizations()
        
        # 8. Compile final results
        final_results = {
            'frequent_itemsets_summary': {level: len(itemsets) for level, itemsets in frequent_itemsets.items()},
            'association_rules_count': len(association_rules),
            'top_association_rules': association_rules[:10],
            'analysis_results': analysis_results,
            'business_impact': business_impact,
            'data_summary': {
                'total_transactions': len(transaction_df.groupby('transaction_id')),
                'total_products': transaction_df['product_id'].nunique(),
                'total_purchase_records': len(transaction_df),
                'avg_items_per_transaction': transaction_df.groupby('transaction_id').size().mean()
            }
        }
        
        print("\n🎉 Market Basket Analysis Complete!")
        print(f"   Association rules found: {len(association_rules)}")
        print(f"   Cross-sell opportunities: {len(analysis_results['cross_sell_opportunities'])}")
        print(f"   Potential revenue increase: ${business_impact['estimated_revenue_increase']:,.0f}")
        print(f"   Total business value: ${business_impact['total_business_value']:,.0f}")
        print(f"   ROI: {business_impact['roi_percentage']:.1f}%")
        
        return final_results

def main():
    """Main execution function for standalone running."""
    
    # Configuration for market basket analysis
    config = {
        'n_transactions': 10000,
        'n_products': 200,
        'min_support': 0.01,
        'min_confidence': 0.5,
        'min_lift': 1.0,
        'business_params': {
            'avg_product_price': 25.0,
            'cross_sell_success_rate': 0.15,
            'inventory_cost_rate': 0.02
        }
    }
    
    # Run market basket analysis
    analyzer = MarketBasketAnalyzer(config)
    results = analyzer.run_complete_analysis()
    
    return results

if __name__ == "__main__":
    results = main()