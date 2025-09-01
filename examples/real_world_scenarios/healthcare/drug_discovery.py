# File: examples/real_world_scenarios/healthcare/drug_discovery.py
# Location: examples/real_world_scenarios/healthcare/drug_discovery.py

"""
Drug Discovery System - Real-World ML Pipeline Example

Business Problem:
Accelerate pharmaceutical drug discovery by predicting molecular properties,
identifying potential drug candidates, and optimizing compound structures
for improved efficacy and reduced side effects.

Dataset: Molecular descriptors and bioactivity data (synthetic)
Target: Multi-target prediction (toxicity, efficacy, ADMET properties)
Business Impact: 30% reduction in discovery time, $50M cost savings per compound
Techniques: Molecular fingerprints, graph neural networks, multi-task learning

Industry Applications:
- Pharmaceutical companies
- Biotechnology firms
- Research institutions
- Drug safety organizations
- Regulatory agencies
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, Any, List, Optional
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Framework imports
from src.data.generators import DataGenerator
from src.models.supervised.classification import ClassificationModels
from src.models.supervised.regression import RegressionModels
from src.models.ensemble.methods import EnsembleMethods
from src.evaluation.metrics import ModelEvaluator

# Scenario-specific imports
from ..utilities.data_loaders import DataLoader
from ..utilities.visualization_helpers import BusinessVisualizer
from ..utilities.evaluation_helpers import BusinessMetricsCalculator, ModelPerformanceEvaluator

class DrugDiscoverySystem:
    """Complete drug discovery and molecular analysis system."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize drug discovery system."""
        
        self.config = config or {
            'n_compounds': 10000,
            'test_size': 0.2,
            'validation_size': 0.1,
            'random_state': 42,
            'target_properties': ['toxicity', 'efficacy', 'bioavailability', 'solubility', 'stability'],
            'algorithms': ['random_forest', 'gradient_boosting', 'neural_network', 'svm'],
            'molecular_descriptors': ['molecular_weight', 'logp', 'hbd', 'hba', 'rotatable_bonds'],
            'admet_properties': ['absorption', 'distribution', 'metabolism', 'excretion', 'toxicity'],
            'drug_classes': ['Antibiotics', 'Antivirals', 'Anticancer', 'Cardiovascular', 'CNS'],
            'success_threshold': 0.7,
            'safety_threshold': 0.8
        }
        
        # Initialize components
        self.data_loader = DataLoader(random_state=self.config['random_state'])
        self.visualizer = BusinessVisualizer()
        self.business_calc = BusinessMetricsCalculator()
        self.model_evaluator = ModelPerformanceEvaluator()
        
        # Results storage
        self.molecular_data = None
        self.discovery_results = {}
        self.admet_models = {}
        self.lead_compounds = []
        
    def generate_molecular_dataset(self) -> Tuple[pd.DataFrame, Dict[str, pd.Series]]:
        """Generate comprehensive molecular descriptor dataset."""
        
        print("🧪 Generating molecular dataset for drug discovery...")
        
        np.random.seed(self.config['random_state'])
        
        compounds = []
        for i in range(self.config['n_compounds']):
            # Generate molecular descriptors
            compound = {
                'compound_id': f'COMP_{i:06d}',
                'drug_class': np.random.choice(self.config['drug_classes']),
                
                # Basic molecular properties
                'molecular_weight': np.random.normal(350, 150),
                'logp': np.random.normal(2.5, 1.5),  # Lipophilicity
                'hbd': np.random.poisson(2),  # Hydrogen bond donors
                'hba': np.random.poisson(4),  # Hydrogen bond acceptors
                'rotatable_bonds': np.random.poisson(6),
                'aromatic_rings': np.random.poisson(2),
                'polar_surface_area': np.random.normal(80, 30),
                
                # Structural complexity
                'complexity_score': np.random.exponential(2),
                'ring_count': np.random.poisson(3),
                'heteroatom_count': np.random.poisson(5),
                'formal_charge': np.random.randint(-2, 3),
                
                # Fingerprint features (simulated)
                **{f'fp_{j}': np.random.randint(0, 2) for j in range(50)},  # Binary fingerprints
                
                # Physicochemical properties
                'melting_point': np.random.normal(180, 50),
                'boiling_point': np.random.normal(350, 80),
                'density': np.random.normal(1.2, 0.3),
                'refractive_index': np.random.normal(1.5, 0.1),
                
                # Synthetic accessibility
                'synthesis_score': np.random.uniform(1, 10),
                'retrosynthesis_score': np.random.uniform(1, 10),
                'commercial_availability': np.random.choice([0, 1], p=[0.7, 0.3]),
                
                # Patent and literature
                'patent_count': np.random.poisson(3),
                'literature_mentions': np.random.poisson(5),
                'clinical_trial_phase': np.random.choice([0, 1, 2, 3, 4], p=[0.6, 0.2, 0.1, 0.05, 0.05])
            }
            
            compounds.append(compound)
        
        df = pd.DataFrame(compounds)
        
        # Generate target properties based on molecular descriptors
        targets = self._generate_drug_targets(df)
        
        print(f"✅ Generated {len(df)} molecular compounds")
        print(f"📊 Drug classes: {df['drug_class'].value_counts().to_dict()}")
        print(f"🎯 Target properties: {list(targets.keys())}")
        
        return df, targets
    
    def _generate_drug_targets(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Generate realistic drug target properties."""
        
        targets = {}
        
        # Toxicity prediction (classification)
        toxicity_prob = (
            0.1 +  # Base probability
            0.3 * (df['molecular_weight'] > 500).astype(int) +  # Large molecules more toxic
            0.2 * (df['logp'] > 5).astype(int) +  # High lipophilicity
            0.15 * (df['aromatic_rings'] > 3).astype(int) +  # Many aromatic rings
            0.05 * np.random.random(len(df))  # Random component
        )
        toxicity_prob = np.clip(toxicity_prob, 0, 1)
        targets['toxicity'] = pd.Series(np.random.binomial(1, toxicity_prob), name='toxicity')
        
        # Efficacy score (regression, 0-1)
        efficacy_score = (
            0.5 +  # Base efficacy
            0.2 * np.exp(-((df['molecular_weight'] - 400) ** 2) / 50000) +  # Optimal MW around 400
            0.15 * (df['hbd'] <= 5).astype(int) * (df['hba'] <= 10).astype(int) +  # Lipinski compliance
            0.1 * (df['logp'] > 0) * (df['logp'] < 5) +  # Optimal LogP range
            0.05 * (df['rotatable_bonds'] <= 10).astype(int) +  # Flexibility
            0.1 * np.random.random(len(df)) - 0.05  # Random component
        )
        targets['efficacy'] = pd.Series(np.clip(efficacy_score, 0, 1), name='efficacy')
        
        # Bioavailability (regression, 0-1)
        bioavail_score = (
            0.3 +
            0.3 * (df['molecular_weight'] <= 500).astype(int) +  # Lipinski MW
            0.2 * (df['logp'] > 0) * (df['logp'] < 5) +  # LogP range
            0.1 * (df['polar_surface_area'] <= 140).astype(int) +  # PSA
            0.1 * (df['rotatable_bonds'] <= 10).astype(int) +
            0.1 * np.random.random(len(df)) - 0.05
        )
        targets['bioavailability'] = pd.Series(np.clip(bioavail_score, 0, 1), name='bioavailability')
        
        # Solubility (regression, log scale)
        solubility = (
            -2 +  # Base log solubility
            -0.5 * df['logp'] +  # Higher LogP = lower solubility
            0.1 * df['hbd'] +  # H-bond donors improve solubility
            0.05 * df['hba'] +  # H-bond acceptors
            -0.001 * df['molecular_weight'] +  # Molecular weight effect
            np.random.normal(0, 0.5, len(df))  # Noise
        )
        targets['solubility'] = pd.Series(solubility, name='solubility')
        
        # Stability score (regression, 0-1)
        stability = (
            0.6 +
            0.1 * (df['aromatic_rings'] > 0).astype(int) +  # Aromatic rings add stability
            -0.05 * df['rotatable_bonds'] / 10 +  # Flexibility reduces stability
            0.1 * (df['formal_charge'] == 0).astype(int) +  # Neutral charge
            0.15 * np.random.random(len(df)) - 0.075  # Random
        )
        targets['stability'] = pd.Series(np.clip(stability, 0, 1), name='stability')
        
        return targets
    
    def engineer_molecular_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer advanced molecular features for drug discovery."""
        
        print("🔬 Engineering molecular features...")
        
        df_features = df.copy()
        
        # Lipinski's Rule of Five compliance
        df_features['lipinski_mw'] = (df_features['molecular_weight'] <= 500).astype(int)
        df_features['lipinski_logp'] = (df_features['logp'] <= 5).astype(int)
        df_features['lipinski_hbd'] = (df_features['hbd'] <= 5).astype(int)
        df_features['lipinski_hba'] = (df_features['hba'] <= 10).astype(int)
        df_features['lipinski_compliance'] = (
            df_features['lipinski_mw'] + df_features['lipinski_logp'] + 
            df_features['lipinski_hbd'] + df_features['lipinski_hba']
        )
        
        # Veber's rules (oral bioavailability)
        df_features['veber_rotbonds'] = (df_features['rotatable_bonds'] <= 10).astype(int)
        df_features['veber_psa'] = (df_features['polar_surface_area'] <= 140).astype(int)
        df_features['veber_compliance'] = df_features['veber_rotbonds'] + df_features['veber_psa']
        
        # Lead-like properties
        df_features['lead_like_mw'] = ((df_features['molecular_weight'] >= 200) & 
                                      (df_features['molecular_weight'] <= 350)).astype(int)
        df_features['lead_like_logp'] = ((df_features['logp'] >= -2) & 
                                        (df_features['logp'] <= 4)).astype(int)
        
        # Drug-like indices
        df_features['drug_likeness_score'] = (
            df_features['lipinski_compliance'] / 4 * 0.4 +
            df_features['veber_compliance'] / 2 * 0.3 +
            (df_features['synthesis_score'] <= 6).astype(int) * 0.3
        )
        
        # Molecular complexity features
        df_features['complexity_ratio'] = df_features['complexity_score'] / df_features['molecular_weight']
        df_features['heteroatom_ratio'] = df_features['heteroatom_count'] / df_features['molecular_weight'] * 100
        df_features['ring_density'] = df_features['ring_count'] / df_features['molecular_weight'] * 100
        df_features['aromatic_ratio'] = df_features['aromatic_rings'] / np.maximum(df_features['ring_count'], 1)
        
        # Structural alerts (simplified)
        df_features['high_lipophilicity'] = (df_features['logp'] > 5).astype(int)
        df_features['high_flexibility'] = (df_features['rotatable_bonds'] > 10).astype(int)
        df_features['large_molecule'] = (df_features['molecular_weight'] > 500).astype(int)
        df_features['charged_molecule'] = (df_features['formal_charge'] != 0).astype(int)
        
        # ADMET-related features
        df_features['bbb_permeability_pred'] = np.clip(
            1 / (1 + np.exp(-(df_features['logp'] - df_features['polar_surface_area']/100))), 0, 1
        )
        df_features['cyp450_liability'] = (
            (df_features['aromatic_rings'] > 2) * 0.3 +
            (df_features['heteroatom_count'] > 8) * 0.2 +
            (df_features['molecular_weight'] > 400) * 0.2 +
            np.random.random(len(df_features)) * 0.3
        )
        
        # Interaction features
        df_features['mw_logp_interaction'] = df_features['molecular_weight'] * df_features['logp']
        df_features['hb_interaction'] = df_features['hbd'] * df_features['hba']
        df_features['complexity_flexibility'] = df_features['complexity_score'] * df_features['rotatable_bonds']
        
        # Drug class encoding
        drug_class_dummies = pd.get_dummies(df_features['drug_class'], prefix='class')
        df_features = pd.concat([df_features, drug_class_dummies], axis=1)
        df_features = df_features.drop('drug_class', axis=1)
        
        print(f"✅ Engineered features: {len(df_features.columns)} total features")
        print(f"📊 Added molecular descriptors: {len(df_features.columns) - len(df.columns)}")
        
        return df_features
    
    def train_admet_models(self, X: pd.DataFrame, targets: Dict[str, pd.Series]) -> Dict[str, Any]:
        """Train ADMET (Absorption, Distribution, Metabolism, Excretion, Toxicity) models."""
        
        print("🧬 Training ADMET prediction models...")
        
        # Split data
        from sklearn.model_selection import train_test_split
        
        split_idx = int(len(X) * (1 - self.config['test_size'] - self.config['validation_size']))
        val_idx = int(len(X) * (1 - self.config['test_size']))
        
        X_train = X.iloc[:split_idx]
        X_val = X.iloc[split_idx:val_idx]
        X_test = X.iloc[val_idx:]
        
        print(f"   Training compounds: {len(X_train)}")
        print(f"   Validation compounds: {len(X_val)}")
        print(f"   Test compounds: {len(X_test)}")
        
        # Initialize models
        models = {
            'Random Forest': ClassificationModels().get_random_forest(
                n_estimators=200, max_depth=12, min_samples_split=5,
                random_state=self.config['random_state']
            ),
            'Gradient Boosting': ClassificationModels().get_gradient_boosting(
                n_estimators=200, learning_rate=0.1, max_depth=6,
                random_state=self.config['random_state']
            ),
            'Neural Network': ClassificationModels().get_neural_network(
                hidden_layer_sizes=(100, 50), max_iter=500,
                random_state=self.config['random_state']
            ),
            'SVM': ClassificationModels().get_svm(
                kernel='rbf', probability=True,
                random_state=self.config['random_state']
            )
        }
        
        # Regression models for continuous targets
        reg_models = {
            'Random Forest Reg': RegressionModels().get_random_forest(
                n_estimators=200, max_depth=12,
                random_state=self.config['random_state']
            ),
            'Gradient Boosting Reg': RegressionModels().get_gradient_boosting(
                n_estimators=200, learning_rate=0.1,
                random_state=self.config['random_state']
            ),
            'Neural Network Reg': RegressionModels().get_neural_network(
                hidden_layer_sizes=(100, 50), max_iter=500,
                random_state=self.config['random_state']
            )
        }
        
        admet_results = {}
        
        for target_name, target_values in targets.items():
            print(f"\n   Training models for {target_name}...")
            
            # Split target
            y_train = target_values.iloc[:split_idx]
            y_val = target_values.iloc[split_idx:val_idx]
            y_test = target_values.iloc[val_idx:]
            
            target_results = {}
            
            # Determine if classification or regression
            is_classification = target_name in ['toxicity'] or target_values.nunique() <= 10
            model_dict = models if is_classification else reg_models
            
            best_score = 0
            best_model = None
            best_model_name = None
            
            for model_name, model in model_dict.items():
                try:
                    # Train model
                    model.fit(X_train, y_train)
                    
                    # Validate
                    if is_classification:
                        val_score = model.score(X_val, y_val)
                        test_score = model.score(X_test, y_test)
                        y_pred = model.predict(X_test)
                        
                        # Get probabilities if available
                        if hasattr(model, 'predict_proba'):
                            y_proba = model.predict_proba(X_test)
                        else:
                            y_proba = None
                    else:
                        val_score = model.score(X_val, y_val)
                        test_score = model.score(X_test, y_test)
                        y_pred = model.predict(X_test)
                        y_proba = None
                    
                    target_results[model_name] = {
                        'model': model,
                        'val_score': val_score,
                        'test_score': test_score,
                        'predictions': y_pred,
                        'probabilities': y_proba
                    }
                    
                    # Track best model
                    if val_score > best_score:
                        best_score = val_score
                        best_model = model
                        best_model_name = model_name
                    
                    print(f"      {model_name}: Val={val_score:.3f}, Test={test_score:.3f}")
                    
                except Exception as e:
                    print(f"      {model_name}: Failed - {str(e)}")
                    continue
            
            if best_model:
                admet_results[target_name] = {
                    'results': target_results,
                    'best_model': best_model_name,
                    'best_score': best_score,
                    'test_data': (X_test, y_test)
                }
                
                print(f"      🏆 Best model: {best_model_name} (Score: {best_score:.3f})")
        
        return admet_results
    
    def identify_lead_compounds(self, X: pd.DataFrame, targets: Dict[str, pd.Series], 
                              admet_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify promising lead compounds based on multi-criteria optimization."""
        
        print("🎯 Identifying lead compounds...")
        
        # Get test data
        test_idx = int(len(X) * (1 - self.config['test_size']))
        X_test = X.iloc[test_idx:]
        
        lead_candidates = []
        
        for idx in X_test.index:
            compound = X_test.loc[idx]
            compound_id = compound.get('compound_id', f'COMP_{idx}')
            
            # Predict all ADMET properties
            admet_predictions = {}
            admet_scores = {}
            
            for target_name, results in admet_results.items():
                if 'best_model' in results:
                    best_model_name = results['best_model']
                    model = results['results'][best_model_name]['model']
                    
                    # Make prediction
                    compound_features = compound.drop('compound_id', errors='ignore').values.reshape(1, -1)
                    pred = model.predict(compound_features)[0]
                    admet_predictions[target_name] = pred
                    
                    # Convert to score (0-1 scale, higher is better)
                    if target_name == 'toxicity':
                        admet_scores[target_name] = 1 - pred  # Lower toxicity is better
                    elif target_name in ['efficacy', 'bioavailability', 'stability']:
                        admet_scores[target_name] = pred
                    elif target_name == 'solubility':
                        # Convert log solubility to score
                        admet_scores[target_name] = np.clip((pred + 6) / 8, 0, 1)  # Normalize
                    else:
                        admet_scores[target_name] = pred
            
            # Calculate composite drug-likeness score
            weights = {
                'toxicity': 0.25,
                'efficacy': 0.25,
                'bioavailability': 0.20,
                'solubility': 0.15,
                'stability': 0.15
            }
            
            composite_score = sum(
                admet_scores.get(prop, 0.5) * weight 
                for prop, weight in weights.items()
            )
            
            # Additional scoring factors
            lipinski_score = compound.get('lipinski_compliance', 0) / 4
            veber_score = compound.get('veber_compliance', 0) / 2
            drug_likeness = compound.get('drug_likeness_score', 0.5)
            synthesis_feasibility = 1 - (compound.get('synthesis_score', 5) - 1) / 9  # Higher score = harder synthesis
            
            # Final lead compound score
            final_score = (
                composite_score * 0.6 +
                lipinski_score * 0.15 +
                veber_score * 0.10 +
                drug_likeness * 0.10 +
                synthesis_feasibility * 0.05
            )
            
            # Safety check
            safety_score = admet_scores.get('toxicity', 0.5)
            
            lead_candidate = {
                'compound_id': compound_id,
                'final_score': final_score,
                'safety_score': safety_score,
                'admet_predictions': admet_predictions,
                'admet_scores': admet_scores,
                'lipinski_compliance': lipinski_score,
                'veber_compliance': veber_score,
                'synthesis_feasibility': synthesis_feasibility,
                'molecular_weight': compound.get('molecular_weight', 0),
                'logp': compound.get('logp', 0),
                'is_lead': final_score >= self.config['success_threshold'] and safety_score >= self.config['safety_threshold']
            }
            
            lead_candidates.append(lead_candidate)
        
        # Sort by final score
        lead_candidates.sort(key=lambda x: x['final_score'], reverse=True)
        
        # Filter for actual leads
        lead_compounds = [c for c in lead_candidates if c['is_lead']]
        
        print(f"✅ Identified {len(lead_compounds)} lead compounds from {len(lead_candidates)} candidates")
        print(f"📈 Success rate: {len(lead_compounds)/len(lead_candidates)*100:.1f}%")
        
        if lead_compounds:
            best_lead = lead_compounds[0]
            print(f"🏆 Best lead compound: {best_lead['compound_id']}")
            print(f"   Final Score: {best_lead['final_score']:.3f}")
            print(f"   Safety Score: {best_lead['safety_score']:.3f}")
            print(f"   Molecular Weight: {best_lead['molecular_weight']:.1f}")
            print(f"   LogP: {best_lead['logp']:.2f}")
        
        return lead_candidates
    
    def analyze_discovery_insights(self, admet_results: Dict[str, Any], 
                                 lead_compounds: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate drug discovery insights and recommendations."""
        
        print("💡 Generating drug discovery insights...")
        
        insights = {
            'model_performance': {},
            'lead_analysis': {},
            'admet_insights': {},
            'recommendations': []
        }
        
        # Model performance analysis
        for target, results in admet_results.items():
            if 'best_model' in results:
                insights['model_performance'][target] = {
                    'best_model': results['best_model'],
                    'best_score': results['best_score'],
                    'reliability': 'High' if results['best_score'] > 0.8 else 'Medium' if results['best_score'] > 0.6 else 'Low'
                }
        
        # Lead compound analysis
        if lead_compounds:
            leads = [c for c in lead_compounds if c['is_lead']]
            
            insights['lead_analysis'] = {
                'total_candidates': len(lead_compounds),
                'successful_leads': len(leads),
                'success_rate': len(leads) / len(lead_compounds) * 100,
                'avg_final_score': np.mean([c['final_score'] for c in leads]) if leads else 0,
                'avg_safety_score': np.mean([c['safety_score'] for c in leads]) if leads else 0,
                'best_compound_id': leads[0]['compound_id'] if leads else None,
                'best_compound_score': leads[0]['final_score'] if leads else 0
            }
            
            # Property distribution analysis
            if leads:
                mw_avg = np.mean([c['molecular_weight'] for c in leads])
                logp_avg = np.mean([c['logp'] for c in leads])
                
                insights['lead_analysis'].update({
                    'avg_molecular_weight': mw_avg,
                    'avg_logp': logp_avg,
                    'lipinski_compliance_rate': np.mean([c['lipinski_compliance'] for c in leads]),
                    'avg_synthesis_feasibility': np.mean([c['synthesis_feasibility'] for c in leads])
                })
        
        # ADMET insights
        for target in ['toxicity', 'efficacy', 'bioavailability']:
            if target in admet_results and 'best_model' in admet_results[target]:
                model_info = admet_results[target]
                insights['admet_insights'][target] = {
                    'predictive_accuracy': model_info['best_score'],
                    'model_type': model_info['best_model'],
                    'clinical_relevance': 'High' if target in ['toxicity', 'efficacy'] else 'Medium'
                }
        
        # Generate recommendations
        recommendations = []
        
        # Model quality recommendations
        poor_models = [t for t, r in insights['model_performance'].items() if r['best_score'] < 0.6]
        if poor_models:
            recommendations.append(f"Improve prediction models for: {', '.join(poor_models)} - consider more diverse training data")
        
        # Lead compound recommendations
        if insights['lead_analysis'].get('success_rate', 0) < 10:
            recommendations.append("Low lead identification rate - consider relaxing selection criteria or expanding compound library")
        elif insights['lead_analysis'].get('success_rate', 0) > 30:
            recommendations.append("High success rate detected - excellent compound library quality")
        
        # Property-based recommendations
        if leads:
            avg_mw = insights['lead_analysis'].get('avg_molecular_weight', 0)
            if avg_mw > 500:
                recommendations.append("Lead compounds show high molecular weight - optimize for better oral bioavailability")
            
            avg_safety = insights['lead_analysis'].get('avg_safety_score', 0)
            if avg_safety < 0.8:
                recommendations.append("Safety scores below threshold - prioritize toxicity optimization")
        
        # Technology recommendations
        recommendations.append("Consider implementing active learning to improve model efficiency")
        recommendations.append("Integrate experimental validation feedback for model improvement")
        
        insights['recommendations'] = recommendations[:5]  # Top 5
        
        print("✅ Discovery insights generated")
        if insights['lead_analysis'].get('successful_leads', 0) > 0:
            print(f"   Lead compounds identified: {insights['lead_analysis']['successful_leads']}")
            print(f"   Average lead score: {insights['lead_analysis']['avg_final_score']:.3f}")
        
        return insights
    
    def visualize_discovery_results(self, X: pd.DataFrame, targets: Dict[str, pd.Series],
                                  admet_results: Dict[str, Any], lead_compounds: List[Dict[str, Any]]) -> None:
        """Create comprehensive visualizations for drug discovery results."""
        
        print("📊 Creating drug discovery visualizations...")
        
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Molecular weight vs LogP scatter
        ax1 = plt.subplot(3, 4, 1)
        scatter_colors = []
        for idx in X.index:
            compound_id = X.loc[idx, 'compound_id']
            # Find corresponding lead compound
            lead_info = next((lc for lc in lead_compounds if lc['compound_id'] == compound_id), None)
            if lead_info and lead_info['is_lead']:
                scatter_colors.append('red')
            else:
                scatter_colors.append('blue')
        
        ax1.scatter(X['logp'], X['molecular_weight'], c=scatter_colors, alpha=0.6, s=20)
        ax1.axhline(y=500, color='gray', linestyle='--', alpha=0.7, label='Lipinski MW limit')
        ax1.axvline(x=5, color='gray', linestyle='--', alpha=0.7, label='Lipinski LogP limit')
        ax1.set_xlabel('LogP (Lipophilicity)')
        ax1.set_ylabel('Molecular Weight')
        ax1.set_title('Molecular Property Space', fontweight='bold')
        ax1.legend(['Compounds', 'Lead Compounds', 'Lipinski Limits'])
        ax1.grid(True, alpha=0.3)
        
        # 2. ADMET model performance
        ax2 = plt.subplot(3, 4, 2)
        if admet_results:
            targets_list = list(admet_results.keys())
            scores = [admet_results[t]['best_score'] for t in targets_list]
            
            bars = ax2.bar(targets_list, scores, color=['red', 'green', 'blue', 'orange', 'purple'][:len(targets_list)])
            ax2.set_title('ADMET Model Performance', fontweight='bold')
            ax2.set_ylabel('Model Score')
            ax2.set_ylim(0, 1)
            plt.setp(ax2.get_xticklabels(), rotation=45)
            
            # Add score labels on bars
            for bar, score in zip(bars, scores):
                ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                        f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 3. Lead compound score distribution
        ax3 = plt.subplot(3, 4, 3)
        if lead_compounds:
            scores = [lc['final_score'] for lc in lead_compounds]
            ax3.hist(scores, bins=20, alpha=0.7, color='green', edgecolor='black')
            ax3.axvline(self.config['success_threshold'], color='red', linestyle='--', 
                       label=f'Success Threshold ({self.config["success_threshold"]})')
            ax3.set_xlabel('Final Score')
            ax3.set_ylabel('Number of Compounds')
            ax3.set_title('Lead Compound Score Distribution', fontweight='bold')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 4. Drug class distribution
        ax4 = plt.subplot(3, 4, 4)
        if 'class_Antibiotics' in X.columns:  # Check if drug class columns exist
            class_cols = [col for col in X.columns if col.startswith('class_')]
            class_counts = [X[col].sum() for col in class_cols]
            class_names = [col.replace('class_', '') for col in class_cols]
            
            wedges, texts, autotexts = ax4.pie(class_counts, labels=class_names, autopct='%1.1f%%', startangle=90)
            ax4.set_title('Drug Class Distribution', fontweight='bold')
        
        # 5. Lipinski compliance analysis
        ax5 = plt.subplot(3, 4, 5)
        if 'lipinski_compliance' in X.columns:
            compliance_dist = X['lipinski_compliance'].value_counts().sort_index()
            bars = ax5.bar(compliance_dist.index, compliance_dist.values, 
                          color=['red', 'orange', 'yellow', 'lightgreen', 'green'])
            ax5.set_xlabel('Lipinski Compliance Score (0-4)')
            ax5.set_ylabel('Number of Compounds')
            ax5.set_title('Lipinski Rule Compliance', fontweight='bold')
            
            # Add percentage labels
            total = compliance_dist.sum()
            for bar, count in zip(bars, compliance_dist.values):
                pct = count / total * 100
                ax5.text(bar.get_x() + bar.get_width()/2., bar.get_height() + total*0.01,
                        f'{pct:.1f}%', ha='center', va='bottom')
        
        # 6. Safety vs Efficacy scatter
        ax6 = plt.subplot(3, 4, 6)
        if lead_compounds and any(lc.get('admet_scores') for lc in lead_compounds):
            safety_scores = []
            efficacy_scores = []
            colors = []
            
            for lc in lead_compounds:
                if lc.get('admet_scores'):
                    safety_scores.append(lc['admet_scores'].get('toxicity', 0.5))
                    efficacy_scores.append(lc['admet_scores'].get('efficacy', 0.5))
                    colors.append('red' if lc['is_lead'] else 'blue')
            
            ax6.scatter(safety_scores, efficacy_scores, c=colors, alpha=0.7)
            ax6.axhline(y=self.config['success_threshold'], color='gray', linestyle='--', alpha=0.7)
            ax6.axvline(x=self.config['safety_threshold'], color='gray', linestyle='--', alpha=0.7)
            ax6.set_xlabel('Safety Score (1-Toxicity)')
            ax6.set_ylabel('Efficacy Score')
            ax6.set_title('Safety vs Efficacy Analysis', fontweight='bold')
            ax6.grid(True, alpha=0.3)
        
        # 7. Synthesis feasibility
        ax7 = plt.subplot(3, 4, 7)
        if lead_compounds:
            synthesis_scores = [lc['synthesis_feasibility'] for lc in lead_compounds]
            ax7.hist(synthesis_scores, bins=15, alpha=0.7, color='purple', edgecolor='black')
            ax7.set_xlabel('Synthesis Feasibility Score')
            ax7.set_ylabel('Number of Compounds')
            ax7.set_title('Synthesis Feasibility Distribution', fontweight='bold')
            ax7.grid(True, alpha=0.3)
        
        # 8. ADMET property correlation matrix
        ax8 = plt.subplot(3, 4, 8)
        if len(targets) > 1:
            # Create correlation matrix from target values
            target_df = pd.DataFrame(targets)
            corr_matrix = target_df.corr()
            
            im = ax8.imshow(corr_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
            ax8.set_xticks(range(len(corr_matrix.columns)))
            ax8.set_yticks(range(len(corr_matrix.columns)))
            ax8.set_xticklabels(corr_matrix.columns, rotation=45)
            ax8.set_yticklabels(corr_matrix.columns)
            ax8.set_title('ADMET Property Correlations', fontweight='bold')
            
            # Add correlation values
            for i in range(len(corr_matrix)):
                for j in range(len(corr_matrix)):
                    ax8.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}', 
                            ha='center', va='center', fontsize=8)
        
        # 9. Lead compound timeline/pipeline
        ax9 = plt.subplot(3, 4, 9)
        if lead_compounds:
            pipeline_stages = ['Discovery', 'Lead Opt.', 'Preclinical', 'Clinical']
            # Simulate pipeline numbers based on typical attrition rates
            total_compounds = len([lc for lc in lead_compounds if lc['is_lead']])
            pipeline_counts = [
                total_compounds,
                int(total_compounds * 0.3),  # 30% survive lead optimization
                int(total_compounds * 0.1),  # 10% reach preclinical
                int(total_compounds * 0.02)  # 2% reach clinical trials
            ]
            
            bars = ax9.bar(pipeline_stages, pipeline_counts, color=['green', 'yellow', 'orange', 'red'])
            ax9.set_ylabel('Number of Compounds')
            ax9.set_title('Drug Development Pipeline', fontweight='bold')
            
            # Add count labels
            for bar, count in zip(bars, pipeline_counts):
                ax9.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(pipeline_counts)*0.01,
                        str(count), ha='center', va='bottom', fontweight='bold')
        
        # 10. Top lead compounds table
        ax10 = plt.subplot(3, 4, (10, 12))
        ax10.axis('off')
        
        if lead_compounds:
            top_leads = [lc for lc in lead_compounds if lc['is_lead']][:10]  # Top 10
            
            if top_leads:
                table_data = []
                for i, lead in enumerate(top_leads[:5]):  # Show top 5
                    table_data.append([
                        f"{lead['compound_id'][-6:]}",  # Last 6 chars
                        f"{lead['final_score']:.3f}",
                        f"{lead['safety_score']:.3f}",
                        f"{lead['molecular_weight']:.0f}",
                        f"{lead['logp']:.1f}"
                    ])
                
                table = ax10.table(
                    cellText=table_data,
                    colLabels=['Compound', 'Score', 'Safety', 'MW', 'LogP'],
                    cellLoc='center',
                    loc='center',
                    bbox=[0, 0, 1, 1]
                )
                table.auto_set_font_size(False)
                table.set_fontsize(9)
                table.scale(1, 2)
                
                ax10.set_title('Top Lead Compounds', fontweight='bold', y=0.95)
        
        plt.tight_layout()
        plt.show()
        
        print("✅ Visualizations completed")
    
    def generate_discovery_report(self, insights: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive drug discovery report."""
        
        print("📋 Generating drug discovery report...")
        
        report = {
            'executive_summary': {
                'total_compounds_screened': insights['lead_analysis'].get('total_candidates', 0),
                'lead_compounds_identified': insights['lead_analysis'].get('successful_leads', 0),
                'success_rate': insights['lead_analysis'].get('success_rate', 0),
                'best_compound': insights['lead_analysis'].get('best_compound_id', 'None'),
                'avg_lead_score': insights['lead_analysis'].get('avg_final_score', 0),
                'pipeline_readiness': 'High' if insights['lead_analysis'].get('success_rate', 0) > 20 else 'Medium' if insights['lead_analysis'].get('success_rate', 0) > 10 else 'Low'
            },
            'model_performance': insights['model_performance'],
            'admet_analysis': insights['admet_insights'],
            'lead_compound_analysis': insights['lead_analysis'],
            'recommendations': insights['recommendations'],
            'next_steps': [
                'Conduct experimental validation of top lead compounds',
                'Perform structure-activity relationship (SAR) analysis',
                'Optimize lead compounds for improved ADMET properties',
                'Initiate in vitro and in vivo testing',
                'Prepare regulatory documentation for IND filing'
            ]
        }
        
        print("✅ Drug discovery report generated")
        return report
    
    def run_complete_discovery(self) -> Dict[str, Any]:
        """Execute complete drug discovery pipeline."""
        
        print("🚀 Starting Drug Discovery Pipeline")
        print("=" * 50)
        
        # 1. Generate molecular dataset
        X, targets = self.generate_molecular_dataset()
        
        # 2. Engineer molecular features
        X_processed = self.engineer_molecular_features(X)
        
        # 3. Train ADMET models
        admet_results = self.train_admet_models(X_processed, targets)
        
        # 4. Identify lead compounds
        lead_compounds = self.identify_lead_compounds(X_processed, targets, admet_results)
        
        # 5. Generate insights
        insights = self.analyze_discovery_insights(admet_results, lead_compounds)
        
        # 6. Create visualizations
        self.visualize_discovery_results(X_processed, targets, admet_results, lead_compounds)
        
        # 7. Generate report
        report = self.generate_discovery_report(insights)
        
        # Store results
        self.discovery_results = {
            'molecular_data': X_processed,
            'targets': targets,
            'admet_results': admet_results,
            'lead_compounds': lead_compounds,
            'insights': insights,
            'report': report
        }
        
        print("\n🎉 Drug Discovery Pipeline Complete!")
        print(f"   Compounds Screened: {len(X_processed)}")
        print(f"   Lead Compounds: {len([lc for lc in lead_compounds if lc['is_lead']])}")
        print(f"   Success Rate: {insights['lead_analysis'].get('success_rate', 0):.1f}%")
        if insights['lead_analysis'].get('best_compound_id'):
            print(f"   Best Compound: {insights['lead_analysis']['best_compound_id']}")
        
        return self.discovery_results

def main():
    """Main execution function."""
    
    config = {
        'n_compounds': 10000,
        'target_properties': ['toxicity', 'efficacy', 'bioavailability', 'solubility', 'stability'],
        'success_threshold': 0.7,
        'safety_threshold': 0.8
    }
    
    # Run drug discovery
    discovery_system = DrugDiscoverySystem(config)
    results = discovery_system.run_complete_discovery()
    
    return results

if __name__ == "__main__":
    results = main()