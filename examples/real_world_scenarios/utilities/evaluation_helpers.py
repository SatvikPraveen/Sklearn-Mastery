# File: examples/real_world_scenarios/utilities/evaluation_helpers.py
# Location: examples/real_world_scenarios/utilities/evaluation_helpers.py

"""
Evaluation Helper Functions for Real-World Scenarios

Provides business-focused evaluation metrics, ROI calculations, and impact analysis
for all real-world ML scenarios.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

class BusinessMetricsCalculator:
    """Calculate business-focused metrics and ROI for ML models."""
    
    def __init__(self):
        self.currency_symbol = "$"
        
    def calculate_churn_business_impact(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                      y_proba: np.ndarray = None,
                                      avg_customer_value: float = 1200,
                                      retention_cost: float = 50,
                                      campaign_cost_per_customer: float = 25) -> Dict[str, float]:
        """Calculate business impact metrics for churn prediction."""
        
        # Basic classification metrics
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Business calculations
        total_customers = len(y_true)
        actual_churners = np.sum(y_true)
        predicted_churners = np.sum(y_pred)
        
        # Revenue calculations
        revenue_saved_from_tp = tp * avg_customer_value * 0.7  # Assume 70% retention success
        revenue_lost_from_fn = fn * avg_customer_value  # Missed churners
        unnecessary_campaign_cost = fp * campaign_cost_per_customer  # False positives
        total_campaign_cost = predicted_churners * campaign_cost_per_customer
        
        # Calculate ROI
        total_benefit = revenue_saved_from_tp
        total_cost = total_campaign_cost + unnecessary_campaign_cost
        net_benefit = total_benefit - total_cost
        roi_percentage = (net_benefit / total_cost) * 100 if total_cost > 0 else 0
        
        # Cost per acquired customer
        cost_per_retained_customer = total_cost / max(tp, 1)
        
        metrics = {
            'total_customers': total_customers,
            'actual_churners': actual_churners,
            'predicted_churners': predicted_churners,
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn,
            'true_negatives': tn,
            'revenue_saved': revenue_saved_from_tp,
            'revenue_lost': revenue_lost_from_fn,
            'campaign_cost': total_campaign_cost,
            'unnecessary_cost': unnecessary_campaign_cost,
            'net_benefit': net_benefit,
            'roi_percentage': roi_percentage,
            'cost_per_retained_customer': cost_per_retained_customer,
            'churn_rate': (actual_churners / total_customers) * 100,
            'precision': tp / max(tp + fp, 1),
            'recall': tp / max(tp + fn, 1),
            'f1_score': f1_score(y_true, y_pred)
        }
        
        return metrics
    
    def calculate_fraud_business_impact(self, y_true: np.ndarray, y_pred: np.ndarray,
                                      transaction_amounts: np.ndarray = None,
                                      avg_fraud_amount: float = 500,
                                      investigation_cost: float = 25,
                                      false_positive_cost: float = 10) -> Dict[str, float]:
        """Calculate business impact metrics for fraud detection."""
        
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Use transaction amounts if provided, otherwise use average
        if transaction_amounts is not None:
            fraud_mask = y_true == 1
            avg_fraud_amount = np.mean(transaction_amounts[fraud_mask]) if np.any(fraud_mask) else avg_fraud_amount
        
        # Business calculations
        fraud_prevented = tp * avg_fraud_amount
        fraud_losses = fn * avg_fraud_amount  # Missed fraudulent transactions
        investigation_costs = (tp + fp) * investigation_cost  # Cost to investigate all alerts
        customer_friction_cost = fp * false_positive_cost  # Cost of false positives
        
        # Calculate savings and ROI
        gross_savings = fraud_prevented
        total_costs = investigation_costs + customer_friction_cost
        net_savings = gross_savings - total_costs
        
        # Detection metrics
        detection_rate = tp / max(tp + fn, 1)
        false_positive_rate = fp / max(fp + tn, 1)
        alert_precision = tp / max(tp + fp, 1)
        
        metrics = {
            'total_transactions': len(y_true),
            'fraud_transactions': np.sum(y_true),
            'fraud_detected': tp,
            'fraud_missed': fn,
            'false_alarms': fp,
            'fraud_prevented_amount': fraud_prevented,
            'fraud_losses': fraud_losses,
            'investigation_costs': investigation_costs,
            'customer_friction_cost': customer_friction_cost,
            'net_savings': net_savings,
            'detection_rate': detection_rate,
            'false_positive_rate': false_positive_rate,
            'alert_precision': alert_precision,
            'cost_per_detection': total_costs / max(tp, 1),
            'savings_rate': (net_savings / fraud_prevented) * 100 if fraud_prevented > 0 else 0
        }
        
        return metrics
    
    def calculate_sales_forecast_impact(self, y_true: np.ndarray, y_pred: np.ndarray,
                                      inventory_cost_per_unit: float = 10,
                                      stockout_cost_per_unit: float = 50,
                                      holding_cost_rate: float = 0.02) -> Dict[str, float]:
        """Calculate business impact for sales forecasting."""
        
        # Forecast errors
        errors = y_pred - y_true
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        mape = np.mean(np.abs(errors / y_true)) * 100
        
        # Over/under forecasting
        overforecast_mask = errors > 0
        underforecast_mask = errors < 0
        
        overforecast_units = np.sum(errors[overforecast_mask])
        underforecast_units = np.abs(np.sum(errors[underforecast_mask]))
        
        # Cost calculations
        excess_inventory_cost = overforecast_units * (inventory_cost_per_unit + holding_cost_rate * inventory_cost_per_unit * 12)
        stockout_cost = underforecast_units * stockout_cost_per_unit
        total_cost = excess_inventory_cost + stockout_cost
        
        # Perfect forecast baseline
        total_actual_sales = np.sum(y_true)
        perfect_forecast_revenue = total_actual_sales * inventory_cost_per_unit
        
        metrics = {
            'mean_absolute_error': mae,
            'mean_squared_error': mse,
            'mean_absolute_percentage_error': mape,
            'r2_score': r2_score(y_true, y_pred),
            'overforecast_units': overforecast_units,
            'underforecast_units': underforecast_units,
            'excess_inventory_cost': excess_inventory_cost,
            'stockout_cost': stockout_cost,
            'total_forecast_cost': total_cost,
            'forecast_accuracy': 100 - mape,
            'cost_per_unit_error': total_cost / max(mae * len(y_true), 1)
        }
        
        return metrics
    
    def calculate_recommendation_business_impact(self, recommendations_made: int,
                                               recommendations_clicked: int,
                                               recommendations_purchased: int,
                                               avg_order_value: float = 75,
                                               recommendation_cost: float = 0.05) -> Dict[str, float]:
        """Calculate business impact for recommendation systems."""
        
        # Basic metrics
        click_through_rate = (recommendations_clicked / max(recommendations_made, 1)) * 100
        conversion_rate = (recommendations_purchased / max(recommendations_clicked, 1)) * 100
        overall_conversion = (recommendations_purchased / max(recommendations_made, 1)) * 100
        
        # Revenue calculations
        total_revenue = recommendations_purchased * avg_order_value
        total_cost = recommendations_made * recommendation_cost
        net_revenue = total_revenue - total_cost
        roi_percentage = (net_revenue / max(total_cost, 1)) * 100
        
        metrics = {
            'recommendations_made': recommendations_made,
            'recommendations_clicked': recommendations_clicked,
            'recommendations_purchased': recommendations_purchased,
            'click_through_rate': click_through_rate,
            'conversion_rate': conversion_rate,
            'overall_conversion_rate': overall_conversion,
            'total_revenue': total_revenue,
            'total_cost': total_cost,
            'net_revenue': net_revenue,
            'roi_percentage': roi_percentage,
            'revenue_per_recommendation': total_revenue / max(recommendations_made, 1),
            'cost_per_acquisition': total_cost / max(recommendations_purchased, 1)
        }
        
        return metrics

class ModelPerformanceEvaluator:
    """Comprehensive model performance evaluation."""
    
    def __init__(self):
        self.classification_metrics = [
            'accuracy', 'precision', 'recall', 'f1', 'auc'
        ]
        self.regression_metrics = [
            'mae', 'mse', 'rmse', 'r2', 'mape'
        ]
    
    def evaluate_classification_model(self, model, X_test: np.ndarray, y_test: np.ndarray,
                                    X_train: np.ndarray = None, y_train: np.ndarray = None,
                                    cv_folds: int = 5) -> Dict[str, Any]:
        """Comprehensive classification model evaluation."""
        
        # Predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Basic metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1': f1_score(y_test, y_pred, average='weighted')
        }
        
        # AUC if probabilities available
        if y_proba is not None:
            metrics['auc'] = roc_auc_score(y_test, y_proba)
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            metrics['roc_data'] = {'fpr': fpr, 'tpr': tpr, 'auc': metrics['auc']}
        
        # Confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(y_test, y_pred)
        
        # Cross-validation if training data provided
        if X_train is not None and y_train is not None:
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='accuracy')
            metrics['cv_scores'] = cv_scores
            metrics['cv_mean'] = cv_scores.mean()
            metrics['cv_std'] = cv_scores.std()
        
        # Feature importance if available
        if hasattr(model, 'feature_importances_'):
            metrics['feature_importance'] = model.feature_importances_
        
        return metrics
    
    def evaluate_regression_model(self, model, X_test: np.ndarray, y_test: np.ndarray,
                                X_train: np.ndarray = None, y_train: np.ndarray = None,
                                cv_folds: int = 5) -> Dict[str, Any]:
        """Comprehensive regression model evaluation."""
        
        y_pred = model.predict(X_test)
        
        # Basic metrics
        metrics = {
            'mae': mean_absolute_error(y_test, y_pred),
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2': r2_score(y_test, y_pred),
            'mape': np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        }
        
        # Cross-validation if training data provided
        if X_train is not None and y_train is not None:
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='neg_mean_squared_error')
            metrics['cv_scores'] = -cv_scores  # Convert to positive
            metrics['cv_mean'] = (-cv_scores).mean()
            metrics['cv_std'] = cv_scores.std()
        
        # Feature importance if available
        if hasattr(model, 'feature_importances_'):
            metrics['feature_importance'] = model.feature_importances_
        
        # Residuals analysis
        residuals = y_test - y_pred
        metrics['residuals'] = {
            'mean': np.mean(residuals),
            'std': np.std(residuals),
            'min': np.min(residuals),
            'max': np.max(residuals)
        }
        
        return metrics
    
    def compare_models(self, models_results: Dict[str, Dict]) -> pd.DataFrame:
        """Compare multiple models and rank them."""
        
        comparison_data = []
        for model_name, results in models_results.items():
            row = {'Model': model_name}
            
            # Add all metrics
            for key, value in results.items():
                if isinstance(value, (int, float)):
                    row[key] = value
                elif key == 'cv_scores':
                    row['CV_Mean'] = np.mean(value)
                    row['CV_Std'] = np.std(value)
            
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        return df
    
    def calculate_model_stability(self, model, X: np.ndarray, y: np.ndarray, 
                                n_iterations: int = 10) -> Dict[str, float]:
        """Calculate model stability across multiple runs."""
        
        scores = []
        for i in range(n_iterations):
            # Add small random noise to simulate real-world variability
            X_noisy = X + np.random.normal(0, 0.01 * np.std(X), X.shape)
            
            model.fit(X_noisy, y)
            score = model.score(X, y)
            scores.append(score)
        
        stability_metrics = {
            'mean_score': np.mean(scores),
            'std_score': np.std(scores),
            'min_score': np.min(scores),
            'max_score': np.max(scores),
            'score_range': np.max(scores) - np.min(scores),
            'coefficient_of_variation': np.std(scores) / np.mean(scores) * 100
        }
        
        return stability_metrics

class BusinessReportGenerator:
    """Generate business-focused reports and summaries."""
    
    def __init__(self):
        self.report_template = """
        {title}
        ================
        
        Executive Summary:
        {executive_summary}
        
        Key Findings:
        {key_findings}
        
        Business Impact:
        {business_impact}
        
        Technical Performance:
        {technical_performance}
        
        Recommendations:
        {recommendations}
        
        Next Steps:
        {next_steps}
        """
    
    def generate_churn_report(self, business_metrics: Dict, model_metrics: Dict) -> str:
        """Generate executive churn analysis report."""
        
        executive_summary = f"""
        • Customer churn rate: {business_metrics['churn_rate']:.1f}%
        • Model accuracy: {model_metrics['accuracy']:.1f}%
        • Projected annual savings: ${business_metrics['net_benefit']*12:,.0f}
        • ROI: {business_metrics['roi_percentage']:.0f}%
        """
        
        key_findings = f"""
        • Model identifies {business_metrics['recall']:.1%} of actual churners
        • {business_metrics['precision']:.1%} of predictions are accurate
        • Cost per retained customer: ${business_metrics['cost_per_retained_customer']:.0f}
        • Revenue saved from prevented churn: ${business_metrics['revenue_saved']:,.0f}
        """
        
        business_impact = f"""
        • Total customers analyzed: {business_metrics['total_customers']:,}
        • Churners prevented: {business_metrics['true_positives']}
        • Revenue protected: ${business_metrics['revenue_saved']:,.0f}
        • Campaign efficiency: {business_metrics['precision']:.1%}
        """
        
        technical_performance = f"""
        • Model precision: {model_metrics['precision']:.3f}
        • Model recall: {model_metrics['recall']:.3f}
        • F1-score: {model_metrics['f1']:.3f}
        • AUC-ROC: {model_metrics.get('auc', 'N/A')}
        """
        
        recommendations = """
        • Focus retention efforts on high-risk segments
        • Optimize campaign targeting to reduce false positives
        • Implement real-time scoring for immediate intervention
        • A/B test retention strategies for different customer segments
        """
        
        next_steps = """
        • Deploy model to production environment
        • Set up monitoring dashboards
        • Schedule monthly model retraining
        • Integrate with CRM for automated campaigns
        """
        
        return self.report_template.format(
            title="Customer Churn Prediction Analysis",
            executive_summary=executive_summary,
            key_findings=key_findings,
            business_impact=business_impact,
            technical_performance=technical_performance,
            recommendations=recommendations,
            next_steps=next_steps
        )
    
    def generate_fraud_report(self, business_metrics: Dict, model_metrics: Dict) -> str:
        """Generate executive fraud detection report."""
        
        executive_summary = f"""
        • Fraud detection rate: {business_metrics['detection_rate']:.1%}
        • False positive rate: {business_metrics['false_positive_rate']:.2%}
        • Annual fraud prevented: ${business_metrics['fraud_prevented_amount']*365:,.0f}
        • Net savings: ${business_metrics['net_savings']:,.0f}
        """
        
        key_findings = f"""
        • Model catches {business_metrics['detection_rate']:.1%} of fraudulent transactions
        • Alert precision: {business_metrics['alert_precision']:.1%}
        • Cost per detection: ${business_metrics['cost_per_detection']:.2f}
        • Customer impact: {business_metrics['false_positive_rate']:.2%} false positive rate
        """
        
        business_impact = f"""
        • Transactions monitored: {business_metrics['total_transactions']:,}
        • Fraud prevented: ${business_metrics['fraud_prevented_amount']:,.0f}
        • Investigation costs: ${business_metrics['investigation_costs']:,.0f}
        • Customer friction cost: ${business_metrics['customer_friction_cost']:,.0f}
        """
        
        recommendations = """
        • Fine-tune model to reduce false positives
        • Implement real-time scoring infrastructure
        • Add behavioral analytics for enhanced detection
        • Create risk-based authentication workflows
        """
        
        return self.report_template.format(
            title="Fraud Detection System Analysis",
            executive_summary=executive_summary,
            key_findings=key_findings,
            business_impact=business_impact,
            technical_performance=f"Model Performance: {model_metrics.get('f1', 'N/A'):.3f} F1-Score",
            recommendations=recommendations,
            next_steps="Deploy to production and monitor performance"
        )

def calculate_lift_and_gain(y_true: np.ndarray, y_proba: np.ndarray, 
                           n_deciles: int = 10) -> Dict[str, np.ndarray]:
    """Calculate lift and gain charts for model evaluation."""
    
    # Create deciles based on probability scores
    df = pd.DataFrame({'actual': y_true, 'predicted_proba': y_proba})
    df['decile'] = pd.qcut(df['predicted_proba'], n_deciles, labels=False, duplicates='drop')
    
    # Calculate metrics by decile
    decile_metrics = df.groupby('decile').agg({
        'actual': ['count', 'sum', 'mean']
    }).round(3)
    
    decile_metrics.columns = ['count', 'positives', 'response_rate']
    decile_metrics = decile_metrics.sort_index(ascending=False)  # Highest probability first
    
    # Calculate cumulative metrics
    decile_metrics['cumulative_positives'] = decile_metrics['positives'].cumsum()
    decile_metrics['cumulative_count'] = decile_metrics['count'].cumsum()
    decile_metrics['cumulative_response_rate'] = (
        decile_metrics['cumulative_positives'] / decile_metrics['cumulative_count']
    )
    
    # Calculate lift and gain
    overall_response_rate = df['actual'].mean()
    decile_metrics['lift'] = decile_metrics['response_rate'] / overall_response_rate
    decile_metrics['gain'] = (
        decile_metrics['cumulative_positives'] / df['actual'].sum() * 100
    )
    
    return {
        'decile_metrics': decile_metrics,
        'lift': decile_metrics['lift'].values,
        'gain': decile_metrics['gain'].values
    }

def calculate_feature_impact(model, X: np.ndarray, feature_names: List[str] = None) -> pd.DataFrame:
    """Calculate and rank feature impact for business interpretation."""
    
    if not hasattr(model, 'feature_importances_'):
        return pd.DataFrame()
    
    importances = model.feature_importances_
    
    if feature_names is None:
        feature_names = [f'Feature_{i}' for i in range(len(importances))]
    
    feature_impact = pd.DataFrame({
        'feature': feature_names,
        'importance': importances,
        'importance_rank': range(1, len(importances) + 1)
    }).sort_values('importance', ascending=False)
    
    # Calculate cumulative importance
    feature_impact['cumulative_importance'] = feature_impact['importance'].cumsum()
    feature_impact['cumulative_percentage'] = (
        feature_impact['cumulative_importance'] / feature_impact['importance'].sum() * 100
    )
    
    return feature_impact.reset_index(drop=True)

def calculate_prediction_intervals(y_pred: np.ndarray, residuals: np.ndarray, 
                                 confidence: float = 0.95) -> Dict[str, np.ndarray]:
    """Calculate prediction intervals for regression models."""
    
    # Calculate prediction standard error
    mse = np.mean(residuals ** 2)
    prediction_std = np.sqrt(mse)
    
    # Calculate confidence intervals
    from scipy.stats import t
    alpha = 1 - confidence
    dof = len(residuals) - 1  # degrees of freedom
    t_value = t.ppf(1 - alpha/2, dof)
    
    margin_of_error = t_value * prediction_std
    
    return {
        'lower_bound': y_pred - margin_of_error,
        'upper_bound': y_pred + margin_of_error,
        'prediction_std': prediction_std,
        'confidence_level': confidence
    }