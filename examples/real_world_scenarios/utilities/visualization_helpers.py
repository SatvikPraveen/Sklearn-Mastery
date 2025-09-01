# File: examples/real_world_scenarios/utilities/visualization_helpers.py
# Location: examples/real_world_scenarios/utilities/visualization_helpers.py

"""
Visualization Utilities for Real-World Scenarios

Provides standardized plotting functions, business dashboards, and executive reports
for all real-world ML scenarios.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class BusinessVisualizer:
    """Executive-level business visualization utilities."""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        self.figsize = figsize
        self.colors = {
            'primary': '#1f77b4',
            'success': '#2ca02c', 
            'warning': '#ff7f0e',
            'danger': '#d62728',
            'info': '#17a2b8'
        }
    
    def plot_churn_analysis_dashboard(self, churn_data: Dict[str, Any], save_path: str = None) -> plt.Figure:
        """Create comprehensive churn analysis dashboard."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Customer Churn Analysis Dashboard', fontsize=20, fontweight='bold')
        
        # 1. Churn Rate by Segment
        if 'churn_by_segment' in churn_data:
            data = churn_data['churn_by_segment']
            axes[0, 0].bar(data.index, data.values, color=self.colors['danger'])
            axes[0, 0].set_title('Churn Rate by Customer Segment')
            axes[0, 0].set_ylabel('Churn Rate (%)')
            axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Monthly Revenue Impact
        if 'revenue_impact' in churn_data:
            data = churn_data['revenue_impact']
            axes[0, 1].plot(data.index, data.values, marker='o', color=self.colors['warning'])
            axes[0, 1].set_title('Monthly Revenue Impact of Churn')
            axes[0, 1].set_ylabel('Revenue Loss ($)')
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Feature Importance
        if 'feature_importance' in churn_data:
            importance = churn_data['feature_importance']
            y_pos = np.arange(len(importance))
            axes[0, 2].barh(y_pos, importance.values, color=self.colors['info'])
            axes[0, 2].set_yticks(y_pos)
            axes[0, 2].set_yticklabels(importance.index)
            axes[0, 2].set_title('Top Churn Risk Factors')
            axes[0, 2].set_xlabel('Importance Score')
        
        # 4. ROC Curve
        if 'roc_data' in churn_data:
            fpr, tpr, auc = churn_data['roc_data']['fpr'], churn_data['roc_data']['tpr'], churn_data['roc_data']['auc']
            axes[1, 0].plot(fpr, tpr, color=self.colors['success'], label=f'AUC = {auc:.3f}')
            axes[1, 0].plot([0, 1], [0, 1], color='gray', linestyle='--')
            axes[1, 0].set_title('Model Performance (ROC Curve)')
            axes[1, 0].set_xlabel('False Positive Rate')
            axes[1, 0].set_ylabel('True Positive Rate')
            axes[1, 0].legend()
        
        # 5. Confusion Matrix
        if 'confusion_matrix' in churn_data:
            cm = churn_data['confusion_matrix']
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1])
            axes[1, 1].set_title('Prediction Accuracy')
            axes[1, 1].set_ylabel('Actual')
            axes[1, 1].set_xlabel('Predicted')
        
        # 6. Business Metrics
        if 'business_metrics' in churn_data:
            metrics = churn_data['business_metrics']
            metric_names = list(metrics.keys())
            metric_values = list(metrics.values())
            
            bars = axes[1, 2].bar(metric_names, metric_values, color=self.colors['primary'])
            axes[1, 2].set_title('Business Impact Metrics')
            axes[1, 2].tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, metric_values):
                height = bar.get_height()
                axes[1, 2].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                               f'{value:,.0f}', ha='center', va='bottom')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_fraud_detection_dashboard(self, fraud_data: Dict[str, Any], save_path: str = None) -> plt.Figure:
        """Create fraud detection monitoring dashboard."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Fraud Detection System Dashboard', fontsize=20, fontweight='bold')
        
        # 1. Real-time Fraud Rate
        if 'hourly_fraud_rate' in fraud_data:
            data = fraud_data['hourly_fraud_rate']
            axes[0, 0].plot(data.index, data.values, marker='o', color=self.colors['danger'])
            axes[0, 0].set_title('Hourly Fraud Rate')
            axes[0, 0].set_ylabel('Fraud Rate (%)')
            axes[0, 0].set_xlabel('Hour of Day')
        
        # 2. Transaction Volume vs Fraud
        if 'transaction_analysis' in fraud_data:
            data = fraud_data['transaction_analysis']
            ax1 = axes[0, 1]
            ax2 = ax1.twinx()
            
            ax1.bar(data.index, data['volume'], alpha=0.7, color=self.colors['info'], label='Volume')
            ax2.plot(data.index, data['fraud_rate'], color=self.colors['danger'], marker='o', label='Fraud Rate')
            
            ax1.set_title('Transaction Volume vs Fraud Rate')
            ax1.set_ylabel('Transaction Volume', color=self.colors['info'])
            ax2.set_ylabel('Fraud Rate (%)', color=self.colors['danger'])
            ax1.legend(loc='upper left')
            ax2.legend(loc='upper right')
        
        # 3. Model Performance Metrics
        if 'performance_metrics' in fraud_data:
            metrics = fraud_data['performance_metrics']
            metric_names = list(metrics.keys())
            metric_values = list(metrics.values())
            
            bars = axes[0, 2].bar(metric_names, metric_values, 
                                color=[self.colors['success'] if v > 0.8 else self.colors['warning'] for v in metric_values])
            axes[0, 2].set_title('Model Performance Metrics')
            axes[0, 2].set_ylim(0, 1)
            axes[0, 2].tick_params(axis='x', rotation=45)
            
            for bar, value in zip(bars, metric_values):
                axes[0, 2].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                               f'{value:.3f}', ha='center', va='bottom')
        
        # 4. Amount Distribution
        if 'amount_distribution' in fraud_data:
            fraud_amounts = fraud_data['amount_distribution']['fraud']
            normal_amounts = fraud_data['amount_distribution']['normal']
            
            axes[1, 0].hist(normal_amounts, bins=50, alpha=0.7, label='Normal', color=self.colors['success'])
            axes[1, 0].hist(fraud_amounts, bins=50, alpha=0.7, label='Fraud', color=self.colors['danger'])
            axes[1, 0].set_title('Transaction Amount Distribution')
            axes[1, 0].set_xlabel('Amount ($)')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_yscale('log')
            axes[1, 0].legend()
        
        # 5. Geographic Fraud Heatmap
        if 'geographic_data' in fraud_data:
            geo_data = fraud_data['geographic_data']
            pivot_data = geo_data.pivot_table(values='fraud_rate', index='state', columns='city', fill_value=0)
            sns.heatmap(pivot_data, cmap='Reds', ax=axes[1, 1], cbar_kws={'label': 'Fraud Rate'})
            axes[1, 1].set_title('Geographic Fraud Distribution')
        
        # 6. Cost-Benefit Analysis
        if 'cost_benefit' in fraud_data:
            data = fraud_data['cost_benefit']
            categories = list(data.keys())
            values = list(data.values())
            
            colors_cb = [self.colors['success'] if v > 0 else self.colors['danger'] for v in values]
            bars = axes[1, 2].bar(categories, values, color=colors_cb)
            axes[1, 2].set_title('Cost-Benefit Analysis ($)')
            axes[1, 2].tick_params(axis='x', rotation=45)
            axes[1, 2].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            
            for bar, value in zip(bars, values):
                height = bar.get_height()
                axes[1, 2].text(bar.get_x() + bar.get_width()/2., 
                               height + (height * 0.05 if height > 0 else height * -0.05),
                               f'${value:,.0f}', ha='center', 
                               va='bottom' if height > 0 else 'top')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_sales_forecast_dashboard(self, forecast_data: Dict[str, Any], save_path: str = None) -> plt.Figure:
        """Create sales forecasting dashboard."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Sales Forecasting Dashboard', fontsize=20, fontweight='bold')
        
        # 1. Historical vs Predicted Sales
        if 'time_series' in forecast_data:
            data = forecast_data['time_series']
            axes[0, 0].plot(data['dates'], data['actual'], label='Actual', color=self.colors['primary'])
            axes[0, 0].plot(data['dates'], data['predicted'], label='Predicted', color=self.colors['warning'], linestyle='--')
            axes[0, 0].fill_between(data['dates'], data['lower_bound'], data['upper_bound'], 
                                   alpha=0.3, color=self.colors['warning'], label='Confidence Interval')
            axes[0, 0].set_title('Sales Forecast vs Actual')
            axes[0, 0].set_ylabel('Sales ($)')
            axes[0, 0].legend()
            axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Forecast Accuracy Metrics
        if 'accuracy_metrics' in forecast_data:
            metrics = forecast_data['accuracy_metrics']
            metric_names = list(metrics.keys())
            metric_values = list(metrics.values())
            
            bars = axes[0, 1].bar(metric_names, metric_values, color=self.colors['success'])
            axes[0, 1].set_title('Forecast Accuracy Metrics')
            axes[0, 1].tick_params(axis='x', rotation=45)
            
            for bar, value in zip(bars, metric_values):
                axes[0, 1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + bar.get_height()*0.01,
                               f'{value:.3f}', ha='center', va='bottom')
        
        # 3. Seasonal Patterns
        if 'seasonal_decomposition' in forecast_data:
            data = forecast_data['seasonal_decomposition']
            axes[1, 0].plot(data['trend'], label='Trend', color=self.colors['primary'])
            axes[1, 0].plot(data['seasonal'], label='Seasonal', color=self.colors['success'])
            axes[1, 0].set_title('Trend and Seasonal Components')
            axes[1, 0].legend()
        
        # 4. Revenue Impact Analysis
        if 'revenue_impact' in forecast_data:
            data = forecast_data['revenue_impact']
            scenarios = list(data.keys())
            revenues = list(data.values())
            
            colors_impact = [self.colors['success'] if 'optimistic' in s.lower() 
                           else self.colors['warning'] if 'pessimistic' in s.lower()
                           else self.colors['primary'] for s in scenarios]
            
            bars = axes[1, 1].bar(scenarios, revenues, color=colors_impact)
            axes[1, 1].set_title('Revenue Scenarios')
            axes[1, 1].set_ylabel('Projected Revenue ($)')
            axes[1, 1].tick_params(axis='x', rotation=45)
            
            for bar, value in zip(bars, revenues):
                axes[1, 1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + bar.get_height()*0.01,
                               f'${value:,.0f}', ha='center', va='bottom')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_interactive_dashboard(self, data: Dict[str, Any], dashboard_type: str = 'churn') -> go.Figure:
        """Create interactive Plotly dashboard."""
        if dashboard_type == 'churn':
            return self._create_churn_interactive_dashboard(data)
        elif dashboard_type == 'fraud':
            return self._create_fraud_interactive_dashboard(data)
        elif dashboard_type == 'sales':
            return self._create_sales_interactive_dashboard(data)
        else:
            raise ValueError(f"Unsupported dashboard type: {dashboard_type}")
    
    def _create_churn_interactive_dashboard(self, data: Dict[str, Any]) -> go.Figure:
        """Create interactive churn dashboard with Plotly."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Churn Rate by Segment', 'Monthly Revenue Impact', 
                          'Feature Importance', 'Model Performance'),
            specs=[[{"type": "bar"}, {"type": "scatter"}],
                   [{"type": "bar"}, {"type": "scatter"}]]
        )
        
        # Add traces for each subplot
        if 'churn_by_segment' in data:
            segment_data = data['churn_by_segment']
            fig.add_trace(
                go.Bar(x=list(segment_data.index), y=list(segment_data.values), 
                       name='Churn Rate', marker_color='red'),
                row=1, col=1
            )
        
        if 'revenue_impact' in data:
            revenue_data = data['revenue_impact']
            fig.add_trace(
                go.Scatter(x=list(revenue_data.index), y=list(revenue_data.values),
                          mode='lines+markers', name='Revenue Loss'),
                row=1, col=2
            )
        
        fig.update_layout(height=800, title_text="Customer Churn Analysis Dashboard")
        return fig
    
    def generate_executive_summary_plot(self, metrics: Dict[str, float], 
                                      title: str = "Business Impact Summary",
                                      save_path: str = None) -> plt.Figure:
        """Create executive summary visualization."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(title, fontsize=18, fontweight='bold')
        
        # KPI Cards
        kpi_data = {k: v for k, v in metrics.items() if k.startswith('kpi_')}
        if kpi_data:
            y_pos = np.arange(len(kpi_data))
            values = list(kpi_data.values())
            labels = [k.replace('kpi_', '').title() for k in kpi_data.keys()]
            
            bars = ax1.barh(y_pos, values, color=self.colors['success'])
            ax1.set_yticks(y_pos)
            ax1.set_yticklabels(labels)
            ax1.set_title('Key Performance Indicators')
            
            for bar, value in zip(bars, values):
                ax1.text(bar.get_width() + bar.get_width()*0.01, bar.get_y() + bar.get_height()/2,
                        f'{value:,.0f}', ha='left', va='center')
        
        # ROI Analysis
        roi_data = {k: v for k, v in metrics.items() if 'roi' in k.lower() or 'savings' in k.lower()}
        if roi_data:
            labels = [k.replace('_', ' ').title() for k in roi_data.keys()]
            values = list(roi_data.values())
            
            colors = [self.colors['success'] if v > 0 else self.colors['danger'] for v in values]
            bars = ax2.bar(labels, values, color=colors)
            ax2.set_title('Return on Investment')
            ax2.tick_params(axis='x', rotation=45)
            ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., 
                        height + (height * 0.05 if height > 0 else height * -0.05),
                        f'{value:+.1f}%', ha='center', 
                        va='bottom' if height > 0 else 'top')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig

class ModelVisualizer:
    """Technical model visualization utilities."""
    
    def __init__(self):
        self.colors = plt.cm.Set3(np.linspace(0, 1, 12))
    
    def plot_model_comparison(self, results: Dict[str, Dict], save_path: str = None) -> plt.Figure:
        """Compare multiple models performance."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        models = list(results.keys())
        
        # 1. Accuracy Comparison
        accuracies = [results[model]['accuracy'] for model in models]
        bars1 = ax1.bar(models, accuracies, color=self.colors[:len(models)])
        ax1.set_title('Model Accuracy Comparison')
        ax1.set_ylabel('Accuracy')
        ax1.tick_params(axis='x', rotation=45)
        
        for bar, acc in zip(bars1, accuracies):
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                    f'{acc:.3f}', ha='center', va='bottom')
        
        # 2. Training Time
        if all('training_time' in results[model] for model in models):
            times = [results[model]['training_time'] for model in models]
            bars2 = ax2.bar(models, times, color=self.colors[:len(models)])
            ax2.set_title('Training Time Comparison')
            ax2.set_ylabel('Time (seconds)')
            ax2.tick_params(axis='x', rotation=45)
        
        # 3. Cross-validation scores
        if all('cv_scores' in results[model] for model in models):
            cv_data = [results[model]['cv_scores'] for model in models]
            ax3.boxplot(cv_data, labels=models)
            ax3.set_title('Cross-Validation Score Distribution')
            ax3.set_ylabel('CV Score')
            ax3.tick_params(axis='x', rotation=45)
        
        # 4. Feature Importance (for first model with feature importance)
        for model in models:
            if 'feature_importance' in results[model]:
                importance = results[model]['feature_importance']
                y_pos = np.arange(len(importance))[:10]  # Top 10 features
                ax4.barh(y_pos, list(importance.values())[:10])
                ax4.set_yticks(y_pos)
                ax4.set_yticklabels(list(importance.keys())[:10])
                ax4.set_title(f'Feature Importance ({model})')
                break
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig