# File: examples/real_world_scenarios/utilities/data_loaders.py
# Location: examples/real_world_scenarios/utilities/data_loaders.py

"""
Data Loading Utilities for Real-World Scenarios

Provides standardized data loading, preprocessing, and synthetic data generation
for all real-world scenario examples.
"""

import pandas as pd
import numpy as np
import os
from typing import Tuple, Optional, Dict, Any, Union
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
import warnings

class DataLoader:
    """Centralized data loading utility for real-world scenarios."""
    
    def __init__(self, data_dir: str = "data", random_state: int = 42):
        self.data_dir = data_dir
        self.random_state = random_state
        
    def load_customer_churn_data(self, n_samples: int = 10000) -> Tuple[pd.DataFrame, pd.Series]:
        """Load/generate customer churn dataset."""
        np.random.seed(self.random_state)
        
        # Generate realistic customer features
        data = {
            'customer_id': range(n_samples),
            'tenure_months': np.random.exponential(24, n_samples).astype(int),
            'monthly_charges': np.random.normal(65, 20, n_samples),
            'total_charges': np.random.normal(1500, 800, n_samples),
            'contract_type': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples, p=[0.5, 0.3, 0.2]),
            'payment_method': np.random.choice(['Electronic check', 'Credit card', 'Bank transfer', 'Mailed check'], n_samples, p=[0.4, 0.3, 0.2, 0.1]),
            'internet_service': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples, p=[0.4, 0.4, 0.2]),
            'online_security': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.3, 0.5, 0.2]),
            'tech_support': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.3, 0.5, 0.2]),
            'streaming_tv': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.4, 0.4, 0.2]),
            'paperless_billing': np.random.choice([True, False], n_samples, p=[0.6, 0.4]),
            'senior_citizen': np.random.choice([0, 1], n_samples, p=[0.84, 0.16]),
            'partner': np.random.choice(['Yes', 'No'], n_samples, p=[0.5, 0.5]),
            'dependents': np.random.choice(['Yes', 'No'], n_samples, p=[0.3, 0.7]),
            'phone_service': np.random.choice(['Yes', 'No'], n_samples, p=[0.9, 0.1]),
            'multiple_lines': np.random.choice(['Yes', 'No', 'No phone service'], n_samples, p=[0.5, 0.4, 0.1])
        }
        
        df = pd.DataFrame(data)
        
        # Generate churn target based on realistic patterns
        churn_prob = (
            0.1 +  # Base churn rate
            0.3 * (df['contract_type'] == 'Month-to-month') +
            0.2 * (df['tenure_months'] < 6) +
            0.15 * (df['monthly_charges'] > 80) +
            0.1 * (df['payment_method'] == 'Electronic check') +
            0.05 * df['senior_citizen']
        )
        
        churn = np.random.binomial(1, churn_prob, n_samples)
        
        # Remove customer_id for modeling
        features = df.drop('customer_id', axis=1)
        target = pd.Series(churn, name='churn')
        
        return features, target
    
    def load_fraud_detection_data(self, n_samples: int = 50000) -> Tuple[pd.DataFrame, pd.Series]:
        """Load/generate credit card fraud detection dataset."""
        np.random.seed(self.random_state)
        
        # Generate transaction features
        data = {
            'amount': np.random.lognormal(3, 1.5, n_samples),
            'hour': np.random.randint(0, 24, n_samples),
            'day_of_week': np.random.randint(0, 7, n_samples),
            'merchant_category': np.random.randint(1, 20, n_samples),
            'is_weekend': lambda x: (x >= 5).astype(int),
            'is_night': lambda x: ((x >= 22) | (x <= 6)).astype(int)
        }
        
        # Create DataFrame
        df = pd.DataFrame({
            'amount': data['amount'],
            'hour': data['hour'], 
            'day_of_week': data['day_of_week'],
            'merchant_category': data['merchant_category']
        })
        
        df['is_weekend'] = data['is_weekend'](df['day_of_week'])
        df['is_night'] = data['is_night'](df['hour'])
        
        # Add derived features
        df['amount_log'] = np.log1p(df['amount'])
        df['amount_zscore'] = (df['amount'] - df['amount'].mean()) / df['amount'].std()
        
        # Generate fraud labels (highly imbalanced)
        fraud_prob = (
            0.001 +  # Base fraud rate
            0.01 * (df['amount'] > df['amount'].quantile(0.95)) +
            0.005 * df['is_night'] +
            0.003 * df['is_weekend'] +
            0.002 * (df['merchant_category'].isin([1, 15, 18]))
        )
        
        fraud = np.random.binomial(1, fraud_prob, n_samples)
        
        return df, pd.Series(fraud, name='is_fraud')
    
    def load_sales_forecasting_data(self, n_periods: int = 365) -> Tuple[pd.DataFrame, pd.Series]:
        """Generate time series sales data."""
        np.random.seed(self.random_state)
        
        # Create date range
        dates = pd.date_range(start='2020-01-01', periods=n_periods, freq='D')
        
        # Generate realistic sales pattern
        trend = np.linspace(1000, 1200, n_periods)
        seasonal = 100 * np.sin(2 * np.pi * np.arange(n_periods) / 365)  # Annual
        weekly = 50 * np.sin(2 * np.pi * np.arange(n_periods) / 7)      # Weekly
        noise = np.random.normal(0, 30, n_periods)
        
        sales = trend + seasonal + weekly + noise
        sales = np.maximum(sales, 0)  # Ensure non-negative
        
        # Create feature matrix
        df = pd.DataFrame({
            'date': dates,
            'day_of_week': dates.dayofweek,
            'month': dates.month,
            'quarter': dates.quarter,
            'is_weekend': (dates.dayofweek >= 5).astype(int),
            'is_holiday': np.random.binomial(1, 0.05, n_periods),  # ~5% holidays
            'promotion': np.random.binomial(1, 0.1, n_periods),    # 10% promotion days
            'temperature': np.random.normal(15, 10, n_periods),
            'sales_lag1': np.nan,  # Will be filled
            'sales_lag7': np.nan   # Will be filled
        })
        
        # Add lagged features
        sales_series = pd.Series(sales)
        df['sales_lag1'] = sales_series.shift(1)
        df['sales_lag7'] = sales_series.shift(7)
        
        # Fill initial NaN values
        df['sales_lag1'].fillna(method='bfill', inplace=True)
        df['sales_lag7'].fillna(method='bfill', inplace=True)
        
        return df, pd.Series(sales, name='sales')
    
    def load_customer_segmentation_data(self, n_customers: int = 5000) -> pd.DataFrame:
        """Generate customer segmentation dataset."""
        np.random.seed(self.random_state)
        
        # Generate customer segments with different characteristics
        segment_sizes = [0.4, 0.3, 0.2, 0.1]  # High value, Medium, Low, Churned
        segments = np.random.choice(4, n_customers, p=segment_sizes)
        
        data = []
        for i, segment in enumerate(segments):
            if segment == 0:  # High-value customers
                annual_spend = np.random.normal(5000, 1000)
                frequency = np.random.poisson(50)
                recency = np.random.exponential(30)
                avg_order_value = np.random.normal(100, 20)
            elif segment == 1:  # Medium customers
                annual_spend = np.random.normal(2000, 500)
                frequency = np.random.poisson(20)
                recency = np.random.exponential(60)
                avg_order_value = np.random.normal(50, 15)
            elif segment == 2:  # Low-value customers
                annual_spend = np.random.normal(500, 200)
                frequency = np.random.poisson(5)
                recency = np.random.exponential(120)
                avg_order_value = np.random.normal(25, 10)
            else:  # Churned customers
                annual_spend = np.random.normal(100, 50)
                frequency = np.random.poisson(1)
                recency = np.random.exponential(300)
                avg_order_value = np.random.normal(20, 5)
            
            data.append({
                'customer_id': f'CUST_{i:05d}',
                'annual_spend': max(annual_spend, 0),
                'frequency': max(frequency, 0),
                'recency': max(recency, 1),
                'avg_order_value': max(avg_order_value, 5),
                'true_segment': segment
            })
        
        df = pd.DataFrame(data)
        
        # Add derived RFM features
        df['monetary_score'] = pd.qcut(df['annual_spend'], 5, labels=[1,2,3,4,5]).astype(int)
        df['frequency_score'] = pd.qcut(df['frequency'], 5, labels=[1,2,3,4,5]).astype(int)
        df['recency_score'] = pd.qcut(df['recency'], 5, labels=[5,4,3,2,1]).astype(int)
        df['rfm_score'] = df['recency_score'].astype(str) + df['frequency_score'].astype(str) + df['monetary_score'].astype(str)
        
        return df
    
    def load_recommendation_data(self, n_users: int = 1000, n_items: int = 500, n_interactions: int = 50000) -> pd.DataFrame:
        """Generate recommendation system dataset."""
        np.random.seed(self.random_state)
        
        # Generate user-item interactions with realistic patterns
        interactions = []
        
        for _ in range(n_interactions):
            # Power law distribution for users (some very active)
            user_id = int(np.random.pareto(0.5) * n_users) % n_users
            
            # Power law for items (some very popular)
            item_id = int(np.random.pareto(0.3) * n_items) % n_items
            
            # Rating based on user and item characteristics
            user_bias = np.random.normal(0, 0.5)
            item_bias = np.random.normal(0, 0.5)
            base_rating = 3.5 + user_bias + item_bias
            
            # Add noise and ensure rating is in valid range
            rating = np.clip(base_rating + np.random.normal(0, 0.3), 1, 5)
            
            interactions.append({
                'user_id': user_id,
                'item_id': item_id,
                'rating': round(rating, 1),
                'timestamp': pd.Timestamp.now() - pd.Timedelta(days=np.random.exponential(30))
            })
        
        df = pd.DataFrame(interactions)
        
        # Remove duplicates (keep latest rating)
        df = df.sort_values('timestamp').drop_duplicates(['user_id', 'item_id'], keep='last')
        
        return df
    
    def load_medical_diagnosis_data(self, n_patients: int = 5000) -> Tuple[pd.DataFrame, pd.Series]:
        """Generate medical diagnosis dataset."""
        np.random.seed(self.random_state)
        
        # Generate patient features
        data = {
            'age': np.random.normal(50, 15, n_patients).astype(int),
            'gender': np.random.choice(['M', 'F'], n_patients),
            'bmi': np.random.normal(26, 5, n_patients),
            'blood_pressure_sys': np.random.normal(130, 20, n_patients),
            'blood_pressure_dia': np.random.normal(80, 15, n_patients),
            'cholesterol': np.random.normal(200, 40, n_patients),
            'glucose': np.random.normal(100, 25, n_patients),
            'smoking': np.random.choice([0, 1], n_patients, p=[0.7, 0.3]),
            'family_history': np.random.choice([0, 1], n_patients, p=[0.6, 0.4]),
            'exercise_hours_week': np.random.exponential(3, n_patients),
            'alcohol_units_week': np.random.exponential(5, n_patients)
        }
        
        df = pd.DataFrame(data)
        
        # Ensure realistic bounds
        df['age'] = np.clip(df['age'], 18, 100)
        df['bmi'] = np.clip(df['bmi'], 15, 50)
        df['blood_pressure_sys'] = np.clip(df['blood_pressure_sys'], 90, 200)
        df['blood_pressure_dia'] = np.clip(df['blood_pressure_dia'], 60, 120)
        
        # Generate disease probability based on risk factors
        disease_prob = (
            0.05 +  # Base disease rate
            0.001 * (df['age'] - 40) +  # Age factor
            0.02 * df['smoking'] +
            0.01 * df['family_history'] +
            0.001 * np.maximum(df['bmi'] - 25, 0) +  # Obesity factor
            0.0001 * np.maximum(df['blood_pressure_sys'] - 140, 0)  # Hypertension
        )
        
        disease = np.random.binomial(1, np.clip(disease_prob, 0, 1), n_patients)
        
        return df, pd.Series(disease, name='has_disease')
    
    def save_dataset(self, X: pd.DataFrame, y: pd.Series, filename: str, include_target: bool = True) -> str:
        """Save dataset to CSV file."""
        filepath = os.path.join(self.data_dir, f"{filename}.csv")
        os.makedirs(self.data_dir, exist_ok=True)
        
        if include_target:
            data = X.copy()
            data[y.name] = y
            data.to_csv(filepath, index=False)
        else:
            X.to_csv(filepath, index=False)
        
        return filepath
    
    def load_custom_csv(self, filepath: str, target_column: str = None) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]]:
        """Load custom CSV dataset."""
        df = pd.read_csv(filepath)
        
        if target_column:
            X = df.drop(target_column, axis=1)
            y = df[target_column]
            return X, y
        else:
            return df
    
    def create_train_test_split(self, X: pd.DataFrame, y: pd.Series, 
                               test_size: float = 0.2, 
                               stratify: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Create stratified train-test split."""
        stratify_param = y if stratify else None
        return train_test_split(X, y, test_size=test_size, 
                               stratify=stratify_param, 
                               random_state=self.random_state)