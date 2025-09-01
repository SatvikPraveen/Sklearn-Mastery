# File: examples/real_world_scenarios/healthcare/medical_diagnosis.py
# Location: examples/real_world_scenarios/healthcare/medical_diagnosis.py

"""
Medical Diagnosis System - Real-World ML Pipeline Example

Business Problem:
Assist healthcare professionals in diagnosing diseases based on patient symptoms,
medical history, and diagnostic test results to improve accuracy and speed.

Dataset: Patient medical records with symptoms and diagnoses (synthetic)
Target: Multi-class classification (disease diagnosis)
Business Impact: 15% improvement in diagnostic accuracy, 40% reduction in diagnosis time
Techniques: Feature engineering, ensemble methods, interpretable ML, clinical validation

Industry Applications:
- Hospitals and clinics
- Telemedicine platforms
- Diagnostic imaging centers
- Primary care practices
- Emergency departments
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
from src.models.supervised.classification import ClassificationModels
from src.models.ensemble.methods import EnsembleMethods
from src.evaluation.metrics import ModelEvaluator

# Scenario-specific imports
from ..utilities.data_loaders import DataLoader
from ..utilities.visualization_helpers import BusinessVisualizer
from ..utilities.evaluation_helpers import BusinessMetricsCalculator, ModelPerformanceEvaluator

class MedicalDiagnosisSystem:
    """Complete medical diagnosis assistance system."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize medical diagnosis system."""
        
        self.config = config or {
            'n_patients': 5000,
            'test_size': 0.2,
            'validation_size': 0.1,
            'random_state': 42,
            'target_diseases': ['Diabetes', 'Hypertension', 'Heart Disease', 'Respiratory Infection', 'Healthy'],
            'algorithms': ['random_forest', 'gradient_boosting', 'svm', 'neural_network'],
            'interpretability_required': True,
            'clinical_validation': True,
            'confidence_threshold': 0.8
        }
        
        # Initialize components
        self.data_loader = DataLoader(random_state=self.config['random_state'])
        self.visualizer = BusinessVisualizer()
        self.business_calc = BusinessMetricsCalculator()
        self.model_evaluator = ModelPerformanceEvaluator()
        
        # Results storage
        self.patient_data = None
        self.model_results = {}
        self.clinical_insights = {}
        self.best_model = None
        
    def load_and_analyze_patient_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Load and analyze synthetic patient medical data."""
        
        print("🔄 Loading medical diagnosis dataset...")
        X, y = self.data_loader.load_medical_diagnosis_data(n_patients=self.config['n_patients'])
        
        # Map binary target to multi-class diseases
        np.random.seed(self.config['random_state'])
        disease_labels = []
        
        for i in range(len(y)):
            if y[i] == 1:  # Has disease
                # Assign disease based on patient characteristics
                patient = X.iloc[i]
                
                # Disease probability based on patient profile
                if patient['age'] > 60 and patient['bmi'] > 30:
                    disease = np.random.choice(['Diabetes', 'Heart Disease', 'Hypertension'], 
                                             p=[0.4, 0.35, 0.25])
                elif patient['age'] < 40:
                    disease = np.random.choice(['Respiratory Infection', 'Hypertension'], 
                                             p=[0.7, 0.3])
                elif patient['smoking'] == 1:
                    disease = np.random.choice(['Heart Disease', 'Respiratory Infection'], 
                                             p=[0.6, 0.4])
                else:
                    disease = np.random.choice(self.config['target_diseases'][:-1])  # Exclude 'Healthy'
            else:
                disease = 'Healthy'
            
            disease_labels.append(disease)
        
        y_multiclass = pd.Series(disease_labels, name='diagnosis')
        
        print(f"📊 Dataset shape: {X.shape}")
        print(f"📊 Number of patients: {len(X)}")
        print(f"📊 Features: {list(X.columns)}")
        
        # Disease distribution
        print("\n📈 Disease Distribution:")
        disease_counts = y_multiclass.value_counts()
        for disease, count in disease_counts.items():
            percentage = (count / len(y_multiclass)) * 100
            print(f"   {disease}: {count} patients ({percentage:.1f}%)")
        
        # Patient demographics analysis
        print("\n👥 Patient Demographics:")
        print(f"   Average age: {X['age'].mean():.1f} years")
        print(f"   Age range: {X['age'].min()}-{X['age'].max()} years")
        print(f"   Gender distribution: {X['gender'].value_counts().to_dict()}")
        print(f"   Average BMI: {X['bmi'].mean():.1f}")
        print(f"   Smoking rate: {X['smoking'].mean():.1%}")
        print(f"   Family history rate: {X['family_history'].mean():.1%}")
        
        # Clinical indicators analysis
        print("\n🩺 Clinical Indicators:")
        print(f"   Average systolic BP: {X['blood_pressure_sys'].mean():.0f} mmHg")
        print(f"   Average diastolic BP: {X['blood_pressure_dia'].mean():.0f} mmHg")
        print(f"   Average cholesterol: {X['cholesterol'].mean():.0f} mg/dL")
        print(f"   Average glucose: {X['glucose'].mean():.0f} mg/dL")
        
        # Store for analysis
        self.patient_data = X
        self.diagnosis_labels = y_multiclass
        
        return X, y_multiclass
    
    def engineer_medical_features(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Create medically relevant features."""
        
        print("\n🧬 Engineering medical features...")
        
        X_medical = X.copy()
        
        # 1. Cardiovascular risk indicators
        X_medical['hypertension_indicator'] = ((X_medical['blood_pressure_sys'] > 140) | 
                                              (X_medical['blood_pressure_dia'] > 90)).astype(int)
        
        X_medical['high_cholesterol'] = (X_medical['cholesterol'] > 200).astype(int)
        
        X_medical['cardiovascular_risk_score'] = (
            (X_medical['age'] > 50).astype(int) * 2 +
            X_medical['smoking'] * 3 +
            X_medical['hypertension_indicator'] * 2 +
            X_medical['high_cholesterol'] * 1 +
            X_medical['family_history'] * 2
        )
        
        # 2. Metabolic indicators
        X_medical['obesity'] = (X_medical['bmi'] > 30).astype(int)
        X_medical['diabetes_indicator'] = (X_medical['glucose'] > 126).astype(int)
        
        X_medical['metabolic_syndrome_score'] = (
            X_medical['obesity'] +
            X_medical['diabetes_indicator'] +
            X_medical['hypertension_indicator'] +
            X_medical['high_cholesterol']
        )
        
        # 3. Age-related risk categories
        X_medical['age_category'] = pd.cut(X_medical['age'], 
                                          bins=[0, 30, 50, 65, 100], 
                                          labels=['Young', 'Middle', 'Senior', 'Elderly'])
        age_dummies = pd.get_dummies(X_medical['age_category'], prefix='age')
        X_medical = pd.concat([X_medical, age_dummies], axis=1)
        
        # 4. BMI categories
        X_medical['bmi_category'] = pd.cut(X_medical['bmi'], 
                                          bins=[0, 18.5, 25, 30, 50], 
                                          labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
        bmi_dummies = pd.get_dummies(X_medical['bmi_category'], prefix='bmi')
        X_medical = pd.concat([X_medical, bmi_dummies], axis=1)
        
        # 5. Blood pressure categories
        X_medical['bp_category'] = 'Normal'
        X_medical.loc[X_medical['blood_pressure_sys'] >= 140, 'bp_category'] = 'High'
        X_medical.loc[X_medical['blood_pressure_sys'] < 90, 'bp_category'] = 'Low'
        bp_dummies = pd.get_dummies(X_medical['bp_category'], prefix='bp')
        X_medical = pd.concat([X_medical, bp_dummies], axis=1)
        
        # 6. Interaction features
        X_medical['age_bmi_interaction'] = X_medical['age'] * X_medical['bmi']
        X_medical['smoking_age_interaction'] = X_medical['smoking'] * X_medical['age']
        X_medical['bp_cholesterol_interaction'] = X_medical['blood_pressure_sys'] * X_medical['cholesterol'] / 1000
        
        # 7. Lifestyle risk score
        X_medical['lifestyle_risk'] = (
            X_medical['smoking'] * 3 +
            (X_medical['exercise_hours_week'] < 2).astype(int) * 2 +
            (X_medical['alcohol_units_week'] > 14).astype(int) * 1
        )
        
        # Clean up categorical columns
        X_medical = X_medical.drop(['age_category', 'bmi_category', 'bp_category'], axis=1)
        
        # Handle gender encoding
        X_medical['gender'] = (X_medical['gender'] == 'M').astype(int)
        
        print(f"✅ Created medical features: {X_medical.shape[1]} total features")
        print(f"📊 New features added: {X_medical.shape[1] - X.shape[1]}")
        
        return X_medical, y
    
    def train_diagnostic_models(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Train multiple diagnostic models with clinical validation."""
        
        print("\n🤖 Training diagnostic models...")
        
        # Split data with stratification
        from sklearn.model_selection import train_test_split
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=self.config['test_size'] + self.config['validation_size'],
            random_state=self.config['random_state'], stratify=y
        )
        
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, 
            test_size=self.config['test_size'] / (self.config['test_size'] + self.config['validation_size']),
            random_state=self.config['random_state'], stratify=y_temp
        )
        
        print(f"   Training set: {len(X_train)} patients")
        print(f"   Validation set: {len(X_val)} patients")
        print(f"   Test set: {len(X_test)} patients")
        
        # Initialize models
        models = ClassificationModels()
        ensemble = EnsembleMethods()
        
        # Configure models for medical diagnosis
        algorithms_to_test = {
            'Random Forest': models.get_random_forest(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                class_weight='balanced',
                random_state=self.config['random_state']
            ),
            'Gradient Boosting': models.get_gradient_boosting(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=5,
                random_state=self.config['random_state']
            ),
            'SVM': models.get_svm(
                kernel='rbf',
                probability=True,
                class_weight='balanced',
                random_state=self.config['random_state']
            ),
            'Logistic Regression': models.get_logistic_regression(
                class_weight='balanced',
                random_state=self.config['random_state']
            )
        }
        
        # Add ensemble model
        base_estimators = [
            ('rf', models.get_random_forest(n_estimators=100, random_state=self.config['random_state'])),
            ('gb', models.get_gradient_boosting(n_estimators=100, random_state=self.config['random_state'])),
            ('lr', models.get_logistic_regression(random_state=self.config['random_state']))
        ]
        
        algorithms_to_test['Medical Ensemble'] = ensemble.get_voting_classifier(
            base_estimators, voting='soft'
        )
        
        # Train and evaluate models
        model_results = {}
        
        for name, model in algorithms_to_test.items():
            print(f"   Training {name}...")
            
            # Fit model
            model.fit(X_train, y_train)
            
            # Validate model
            val_performance = self.model_evaluator.evaluate_classification_model(
                model, X_val, y_val, X_train, y_train, cv_folds=3
            )
            
            # Test model
            test_performance = self.model_evaluator.evaluate_classification_model(
                model, X_test, y_test
            )
            
            # Clinical validation
            clinical_metrics = self.perform_clinical_validation(
                model, X_test, y_test, name
            )
            
            # Calculate confidence scores
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test)
                confidence_scores = self.calculate_diagnostic_confidence(y_proba)
            else:
                confidence_scores = {'avg_confidence': 0.5, 'high_confidence_rate': 0.0}
            
            model_results[name] = {
                'model': model,
                'validation_performance': val_performance,
                'test_performance': test_performance,
                'clinical_metrics': clinical_metrics,
                'confidence_scores': confidence_scores
            }
            
            print(f"      Validation Accuracy: {val_performance['accuracy']:.3f}")
            print(f"      Test Accuracy: {test_performance['accuracy']:.3f}")
            print(f"      Clinical Score: {clinical_metrics['clinical_utility_score']:.3f}")
            print(f"      High Confidence Rate: {confidence_scores['high_confidence_rate']:.1%}")
        
        # Select best model based on clinical utility
        best_model_name = max(model_results.keys(), 
                            key=lambda x: model_results[x]['clinical_metrics']['clinical_utility_score'])
        
        self.best_model = model_results[best_model_name]['model']
        self.best_model_name = best_model_name
        
        print(f"\n🏆 Best diagnostic model: {best_model_name}")
        print(f"   Clinical Utility Score: {model_results[best_model_name]['clinical_metrics']['clinical_utility_score']:.3f}")
        print(f"   Test Accuracy: {model_results[best_model_name]['test_performance']['accuracy']:.3f}")
        
        # Store results
        self.model_results = model_results
        self.test_data = (X_test, y_test)
        
        return model_results
    
    def perform_clinical_validation(self, model, X_test: pd.DataFrame, y_test: pd.Series, model_name: str) -> Dict[str, Any]:
        """Perform clinical validation of the diagnostic model."""
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)
            confidence_scores = np.max(y_proba, axis=1)
        else:
            confidence_scores = np.full(len(y_pred), 0.7)  # Default confidence
        
        # Clinical metrics
        from sklearn.metrics import classification_report, confusion_matrix
        
        # Calculate per-disease metrics
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Clinical utility scoring
        disease_severity_weights = {
            'Heart Disease': 1.0,      # High severity
            'Diabetes': 0.8,           # Medium-high severity
            'Hypertension': 0.6,       # Medium severity
            'Respiratory Infection': 0.4,  # Medium-low severity
            'Healthy': 0.1             # Low severity (false positive cost)
        }
        
        # Calculate weighted clinical utility
        clinical_utility_score = 0.0
        total_weight = 0.0
        
        for disease in self.config['target_diseases']:
            if disease in report:
                precision = report[disease]['precision']
                recall = report[disease]['recall']
                weight = disease_severity_weights.get(disease, 0.5)
                
                # F-beta score with higher weight on recall for serious diseases
                beta = 2.0 if weight > 0.7 else 1.0
                f_beta = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall) if (precision + recall) > 0 else 0
                
                clinical_utility_score += f_beta * weight
                total_weight += weight
        
        clinical_utility_score /= total_weight if total_weight > 0 else 1
        
        # Sensitivity analysis for high-risk conditions
        high_risk_diseases = ['Heart Disease', 'Diabetes']
        high_risk_sensitivity = {}
        
        for disease in high_risk_diseases:
            if disease in report:
                high_risk_sensitivity[disease] = report[disease]['recall']
        
        # Diagnostic confidence analysis
        high_confidence_predictions = (confidence_scores > self.config['confidence_threshold']).sum()
        high_confidence_rate = high_confidence_predictions / len(confidence_scores)
        
        # Error analysis
        critical_misses = 0  # Cases where serious disease was missed
        false_alarms = 0     # Cases where healthy patient was flagged
        
        for true_label, pred_label in zip(y_test, y_pred):
            if true_label in high_risk_diseases and pred_label == 'Healthy':
                critical_misses += 1
            elif true_label == 'Healthy' and pred_label in high_risk_diseases:
                false_alarms += 1
        
        clinical_metrics = {
            'clinical_utility_score': clinical_utility_score,
            'high_risk_sensitivity': high_risk_sensitivity,
            'high_confidence_rate': high_confidence_rate,
            'critical_misses': critical_misses,
            'false_alarms': false_alarms,
            'avg_confidence': np.mean(confidence_scores),
            'classification_report': report
        }
        
        return clinical_metrics
    
    def calculate_diagnostic_confidence(self, y_proba: np.ndarray) -> Dict[str, float]:
        """Calculate diagnostic confidence metrics."""
        
        # Maximum probability as confidence
        max_proba = np.max(y_proba, axis=1)
        
        # Entropy-based uncertainty
        epsilon = 1e-10
        entropy = -np.sum(y_proba * np.log(y_proba + epsilon), axis=1)
        max_entropy = np.log(y_proba.shape[1])  # Maximum possible entropy
        uncertainty = entropy / max_entropy
        confidence_from_entropy = 1 - uncertainty
        
        # Combined confidence score
        combined_confidence = (max_proba + confidence_from_entropy) / 2
        
        return {
            'avg_confidence': np.mean(combined_confidence),
            'high_confidence_rate': np.mean(combined_confidence > self.config['confidence_threshold']),
            'confidence_distribution': {
                'low': np.mean(combined_confidence < 0.5),
                'medium': np.mean((combined_confidence >= 0.5) & (combined_confidence < 0.8)),
                'high': np.mean(combined_confidence >= 0.8)
            }
        }
    
    def generate_clinical_insights(self, model_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate clinical insights and interpretability analysis."""
        
        print("\n🔬 Generating clinical insights...")
        
        best_results = model_results[self.best_model_name]
        
        # Feature importance analysis
        clinical_features = {}
        
        if hasattr(self.best_model, 'feature_importances_'):
            feature_names = self.test_data[0].columns
            importance_scores = self.best_model.feature_importances_
            
            # Map features to clinical significance
            feature_importance = dict(zip(feature_names, importance_scores))
            
            # Group by clinical categories
            clinical_categories = {
                'Cardiovascular': ['blood_pressure_sys', 'blood_pressure_dia', 'cholesterol', 'cardiovascular_risk_score'],
                'Metabolic': ['bmi', 'glucose', 'diabetes_indicator', 'metabolic_syndrome_score'],
                'Lifestyle': ['smoking', 'exercise_hours_week', 'alcohol_units_week', 'lifestyle_risk'],
                'Demographics': ['age', 'gender', 'family_history'],
                'Clinical_Indicators': ['hypertension_indicator', 'high_cholesterol', 'obesity']
            }
            
            for category, features in clinical_categories.items():
                category_importance = sum(feature_importance.get(feature, 0) for feature in features if feature in feature_importance)
                clinical_features[category] = category_importance
        
        # Diagnostic patterns analysis
        X_test, y_test = self.test_data
        y_pred = self.best_model.predict(X_test)
        
        diagnostic_patterns = {}
        for disease in self.config['target_diseases']:
            disease_mask = y_test == disease
            if disease_mask.sum() > 0:
                disease_patients = X_test[disease_mask]
                
                patterns = {
                    'avg_age': disease_patients['age'].mean(),
                    'gender_distribution': disease_patients['gender'].mean(),
                    'avg_bmi': disease_patients['bmi'].mean(),
                    'smoking_rate': disease_patients['smoking'].mean(),
                    'family_history_rate': disease_patients['family_history'].mean()
                }
                
                diagnostic_patterns[disease] = patterns
        
        # Risk factor analysis
        risk_factors = self.analyze_risk_factors(X_test, y_test, y_pred)
        
        # Clinical recommendations
        recommendations = self.generate_clinical_recommendations(best_results)
        
        insights = {
            'clinical_feature_importance': clinical_features,
            'diagnostic_patterns': diagnostic_patterns,
            'risk_factors': risk_factors,
            'model_interpretability': {
                'feature_count': len(X_test.columns),
                'interpretable': hasattr(self.best_model, 'feature_importances_'),
                'confidence_threshold': self.config['confidence_threshold']
            },
            'clinical_recommendations': recommendations,
            'validation_summary': {
                'accuracy': best_results['test_performance']['accuracy'],
                'clinical_utility': best_results['clinical_metrics']['clinical_utility_score'],
                'high_confidence_rate': best_results['confidence_scores']['high_confidence_rate']
            }
        }
        
        print("✅ Clinical insights generated")
        print(f"   Top clinical factor: {max(clinical_features.items(), key=lambda x: x[1])[0]}")
        print(f"   Model interpretability: {'High' if insights['model_interpretability']['interpretable'] else 'Limited'}")
        
        return insights
    
    def analyze_risk_factors(self, X_test: pd.DataFrame, y_test: pd.Series, y_pred: pd.Series) -> Dict[str, Any]:
        """Analyze risk factors for each disease."""
        
        risk_analysis = {}
        
        for disease in self.config['target_diseases']:
            if disease == 'Healthy':
                continue
                
            disease_mask = y_test == disease
            healthy_mask = y_test == 'Healthy'
            
            if disease_mask.sum() > 0 and healthy_mask.sum() > 0:
                disease_patients = X_test[disease_mask]
                healthy_patients = X_test[healthy_mask]
                
                # Statistical comparison
                risk_factors = {}
                
                for feature in ['age', 'bmi', 'blood_pressure_sys', 'cholesterol', 'smoking', 'family_history']:
                    if feature in X_test.columns:
                        disease_mean = disease_patients[feature].mean()
                        healthy_mean = healthy_patients[feature].mean()
                        
                        # Calculate relative risk
                        if healthy_mean > 0:
                            relative_risk = disease_mean / healthy_mean
                        else:
                            relative_risk = 1.0
                        
                        risk_factors[feature] = {
                            'disease_avg': disease_mean,
                            'healthy_avg': healthy_mean,
                            'relative_risk': relative_risk,
                            'risk_level': 'High' if relative_risk > 1.5 else 'Medium' if relative_risk > 1.2 else 'Low'
                        }
                
                risk_analysis[disease] = risk_factors
        
        return risk_analysis
    
    def generate_clinical_recommendations(self, best_results: Dict[str, Any]) -> List[str]:
        """Generate clinical recommendations based on model performance."""
        
        recommendations = []
        
        clinical_metrics = best_results['clinical_metrics']
        
        # Accuracy-based recommendations
        if best_results['test_performance']['accuracy'] > 0.85:
            recommendations.append("Model demonstrates high diagnostic accuracy suitable for clinical decision support")
        elif best_results['test_performance']['accuracy'] > 0.75:
            recommendations.append("Model shows good accuracy but should be used with physician oversight")
        else:
            recommendations.append("Model requires further improvement before clinical deployment")
        
        # Confidence-based recommendations
        if clinical_metrics['high_confidence_rate'] > 0.7:
            recommendations.append("High proportion of confident predictions suitable for automated screening")
        else:
            recommendations.append("Consider implementing uncertainty quantification for low-confidence predictions")
        
        # Critical miss analysis
        if clinical_metrics['critical_misses'] > 0:
            recommendations.append(f"Monitor for {clinical_metrics['critical_misses']} critical diagnostic misses - implement safety protocols")
        
        # False alarm analysis
        if clinical_metrics['false_alarms'] > len(self.test_data[1]) * 0.1:
            recommendations.append("High false positive rate may cause patient anxiety - consider adjusting threshold")
        
        # Feature importance recommendations
        recommendations.append("Focus on cardiovascular and metabolic indicators for improved accuracy")
        recommendations.append("Implement regular model retraining with new clinical data")
        
        return recommendations
    
    def run_complete_analysis(self) -> Dict[str, Any]:
        """Execute complete medical diagnosis analysis."""
        
        print("🚀 Starting Medical Diagnosis System Analysis")
        print("=" * 50)
        
        # 1. Load and analyze data
        X, y = self.load_and_analyze_patient_data()
        
        # 2. Engineer medical features
        X_processed, y_processed = self.engineer_medical_features(X, y)
        
        # 3. Train diagnostic models
        model_results = self.train_diagnostic_models(X_processed, y_processed)
        
        # 4. Generate clinical insights
        clinical_insights = self.generate_clinical_insights(model_results)
        
        # 5. Compile final results
        final_results = {
            'best_model': self.best_model_name,
            'model_performance': model_results[self.best_model_name]['test_performance'],
            'clinical_validation': model_results[self.best_model_name]['clinical_metrics'],
            'clinical_insights': clinical_insights,
            'data_summary': {
                'total_patients': len(X),
                'disease_distribution': y.value_counts().to_dict(),
                'features_count': len(X_processed.columns),
                'interpretable_model': hasattr(self.best_model, 'feature_importances_')
            }
        }
        
        print("\n🎉 Medical Diagnosis Analysis Complete!")
        print(f"   Best Model: {self.best_model_name}")
        print(f"   Diagnostic Accuracy: {final_results['model_performance']['accuracy']:.1%}")
        print(f"   Clinical Utility Score: {final_results['clinical_validation']['clinical_utility_score']:.3f}")
        print(f"   High Confidence Rate: {final_results['clinical_validation']['high_confidence_rate']:.1%}")
        
        return final_results

def main():
    """Main execution function for standalone running."""
    
    # Configuration for medical diagnosis
    config = {
        'n_patients': 5000,
        'target_diseases': ['Diabetes', 'Hypertension', 'Heart Disease', 'Respiratory Infection', 'Healthy'],
        'algorithms': ['random_forest', 'gradient_boosting', 'svm', 'ensemble'],
        'confidence_threshold': 0.8,
        'interpretability_required': True
    }
    
    # Run medical diagnosis analysis
    diagnosis_system = MedicalDiagnosisSystem(config)
    results = diagnosis_system.run_complete_analysis()
    
    return results

if __name__ == "__main__":
    results = main()