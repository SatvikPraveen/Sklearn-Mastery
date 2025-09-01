# File: examples/real_world_scenarios/__init__.py
# Location: examples/real_world_scenarios/__init__.py

"""
Real-World ML Scenarios Package

This package contains complete, production-ready examples of machine learning
applications across various industries and use cases.

Categories:
- Business Analytics: Churn, sales, fraud, market analysis
- Healthcare: Diagnosis, drug discovery, patient outcomes
- Finance: Credit scoring, trading, risk assessment
- Technology: Recommendations, anomaly detection, maintenance
- Manufacturing: Quality control, supply chain, demand
- Marketing: Segmentation, campaigns, sentiment analysis

Each example includes:
- Complete end-to-end workflow
- Real or realistic synthetic datasets
- Multiple algorithm comparisons
- Business metrics and ROI analysis
- Production deployment guidance
- Comprehensive documentation
"""

from .utilities.data_loaders import *
from .utilities.visualization_helpers import *
from .utilities.evaluation_helpers import *

__version__ = "1.0.0"
__author__ = "ML Pipeline Framework Team"

# Industry scenario mappings
SCENARIO_CATEGORIES = {
    'business_analytics': [
        'customer_churn_prediction',
        'sales_forecasting', 
        'market_basket_analysis',
        'fraud_detection'
    ],
    'healthcare': [
        'medical_diagnosis',
        'drug_discovery',
        'patient_outcome_prediction'
    ],
    'finance': [
        'credit_scoring',
        'algorithmic_trading',
        'risk_assessment',
        'portfolio_optimization'
    ],
    'technology': [
        'recommendation_systems',
        'anomaly_detection',
        'predictive_maintenance',
        'natural_language_processing'
    ],
    'manufacturing': [
        'quality_control',
        'supply_chain_optimization',
        'demand_forecasting'
    ],
    'marketing': [
        'customer_segmentation',
        'campaign_optimization',
        'sentiment_analysis'
    ]
}

# Difficulty levels
DIFFICULTY_LEVELS = {
    'beginner': ['customer_segmentation', 'sales_forecasting'],
    'intermediate': ['customer_churn_prediction', 'fraud_detection', 'recommendation_systems'],
    'advanced': ['algorithmic_trading', 'drug_discovery', 'natural_language_processing']
}

def list_scenarios(category=None, difficulty=None):
    """List available real-world scenarios by category or difficulty."""
    if category:
        return SCENARIO_CATEGORIES.get(category, [])
    elif difficulty:
        return DIFFICULTY_LEVELS.get(difficulty, [])
    else:
        return SCENARIO_CATEGORIES

def get_scenario_info(scenario_name):
    """Get detailed information about a specific scenario."""
    scenario_info = {
        'customer_churn_prediction': {
            'category': 'business_analytics',
            'difficulty': 'intermediate',
            'business_impact': 'High',
            'techniques': ['Classification', 'Feature Engineering', 'Ensemble Methods'],
            'metrics': ['Precision', 'Recall', 'ROI'],
            'industry': 'Telecommunications, SaaS, Retail'
        },
        'fraud_detection': {
            'category': 'business_analytics',
            'difficulty': 'intermediate',
            'business_impact': 'Critical',
            'techniques': ['Anomaly Detection', 'Imbalanced Learning', 'Real-time Scoring'],
            'metrics': ['Precision', 'Recall', 'False Positive Rate'],
            'industry': 'Banking, E-commerce, Insurance'
        }
        # Add more scenarios as needed
    }
    return scenario_info.get(scenario_name, {})

__all__ = [
    'SCENARIO_CATEGORIES',
    'DIFFICULTY_LEVELS', 
    'list_scenarios',
    'get_scenario_info'
]