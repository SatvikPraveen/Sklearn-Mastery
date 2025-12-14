"""Utility functions and helper methods for the sklearn-mastery project."""

import sys
from pathlib import Path as PathLib

# Handle imports that work in both package and direct import contexts
try:
    from ..config.settings import settings
    from ..config.logging_config import LoggerMixin
except ImportError:
    # Fallback for direct imports outside package context
    sys.path.insert(0, str(PathLib(__file__).parent.parent.parent))
    from config.settings import settings
    from config.logging_config import LoggerMixin

import numpy as np
import pandas as pd
from typing import Union, List, Dict, Any, Tuple, Optional
import pickle
import joblib
from pathlib import Path
import json
import yaml
from datetime import datetime
import warnings

from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


class DataUtils(LoggerMixin):
    """Utility functions for data manipulation and analysis."""
    
    @staticmethod
    def smart_train_test_split(
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        test_size: float = None,
        stratify: bool = True,
        random_state: int = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Smart train-test split with automatic stratification detection.
        
        Args:
            X: Feature matrix.
            y: Target vector.
            test_size: Proportion of test set.
            stratify: Whether to use stratification.
            random_state: Random state for reproducibility.
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        test_size = test_size or settings.DEFAULT_TEST_SIZE
        random_state = random_state or settings.RANDOM_SEED
        
        # Determine if we should stratify
        stratify_target = None
        if stratify:
            # Check if y is suitable for stratification
            unique_values = len(np.unique(y))
            total_samples = len(y)
            
            # Use stratification for classification with reasonable number of classes
            if unique_values <= 20 and unique_values < total_samples * 0.5:
                # Check if all classes have enough samples
                unique, counts = np.unique(y, return_counts=True)
                min_count = np.min(counts)
                
                if min_count >= 2:  # Need at least 2 samples per class
                    stratify_target = y
        
        return train_test_split(
            X, y,
            test_size=test_size,
            stratify=stratify_target,
            random_state=random_state
        )
    
    @staticmethod
    def detect_data_types(df: pd.DataFrame) -> Dict[str, List[str]]:
        """Automatically detect column data types for preprocessing.
        
        Args:
            df: DataFrame to analyze.
            
        Returns:
            Dictionary with lists of column names by type.
        """
        data_types = {
            'numerical': [],
            'categorical': [],
            'ordinal': [],
            'datetime': [],
            'text': [],
            'binary': []
        }
        
        for col in df.columns:
            series = df[col]
            
            # Skip if all null
            if series.isnull().all():
                continue
            
            # Datetime detection
            if pd.api.types.is_datetime64_any_dtype(series):
                data_types['datetime'].append(col)
                continue
            
            # Numerical detection
            if pd.api.types.is_numeric_dtype(series):
                # Check if binary (0/1 or True/False)
                unique_vals = series.dropna().unique()
                if len(unique_vals) == 2 and set(unique_vals).issubset({0, 1, True, False}):
                    data_types['binary'].append(col)
                else:
                    data_types['numerical'].append(col)
                continue
            
            # Categorical/ordinal detection
            if pd.api.types.is_categorical_dtype(series):
                if series.cat.ordered:
                    data_types['ordinal'].append(col)
                else:
                    data_types['categorical'].append(col)
                continue
            
            # String/object type analysis
            if series.dtype == 'object':
                unique_vals = series.dropna().unique()
                unique_count = len(unique_vals)
                total_count = len(series.dropna())
                
                # High cardinality suggests text
                if unique_count > total_count * 0.5:
                    data_types['text'].append(col)
                # Low cardinality suggests categorical
                elif unique_count <= 20:
                    data_types['categorical'].append(col)
                else:
                    # Medium cardinality - could be either
                    # Check average string length
                    avg_length = series.dropna().astype(str).str.len().mean()
                    if avg_length > 20:
                        data_types['text'].append(col)
                    else:
                        data_types['categorical'].append(col)
        
        return data_types
    
    @staticmethod
    def encode_target_variable(
        y: Union[np.ndarray, pd.Series, List],
        return_encoder: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, LabelEncoder]]:
        """Encode target variable for classification tasks.
        
        Args:
            y: Target variable.
            return_encoder: Whether to return the fitted encoder.
            
        Returns:
            Encoded target variable, optionally with encoder.
        """
        y = np.asarray(y)
        
        # Check if already numeric
        if np.issubdtype(y.dtype, np.number):
            if return_encoder:
                return y, None
            return y
        
        # Encode string/categorical targets
        encoder = LabelEncoder()
        y_encoded = encoder.fit_transform(y)
        
        if return_encoder:
            return y_encoded, encoder
        return y_encoded
    
    @staticmethod
    def check_data_quality(
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.Series]] = None
    ) -> Dict[str, Any]:
        """Comprehensive data quality check.
        
        Args:
            X: Feature matrix.
            y: Target vector (optional).
            
        Returns:
            Dictionary with data quality metrics.
        """
        quality_report = {}
        
        # Convert to DataFrame if necessary
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        
        # Basic info
        quality_report['n_samples'] = len(X)
        quality_report['n_features'] = len(X.columns)
        
        # Missing values
        missing_counts = X.isnull().sum()
        quality_report['missing_values'] = {
            'total': missing_counts.sum(),
            'percentage': (missing_counts.sum() / (len(X) * len(X.columns))) * 100,
            'columns_with_missing': missing_counts[missing_counts > 0].to_dict()
        }
        
        # Duplicate rows
        quality_report['duplicate_rows'] = X.duplicated().sum()
        
        # Data types
        quality_report['data_types'] = X.dtypes.value_counts().to_dict()
        
        # Numerical features analysis
        numerical_cols = X.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 0:
            quality_report['numerical_features'] = {
                'count': len(numerical_cols),
                'infinite_values': np.isinf(X[numerical_cols]).sum().sum(),
                'zero_variance': (X[numerical_cols].var() == 0).sum()
            }
        
        # Categorical features analysis
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            cardinalities = X[categorical_cols].nunique()
            quality_report['categorical_features'] = {
                'count': len(categorical_cols),
                'high_cardinality': (cardinalities > 50).sum(),
                'avg_cardinality': cardinalities.mean()
            }
        
        # Target variable analysis (if provided)
        if y is not None:
            y = pd.Series(y) if not isinstance(y, pd.Series) else y
            
            quality_report['target'] = {
                'type': str(y.dtype),
                'missing_values': y.isnull().sum(),
                'unique_values': y.nunique()
            }
            
            # Classification specific
            if y.nunique() <= 20:  # Likely classification
                value_counts = y.value_counts()
                quality_report['target']['class_distribution'] = value_counts.to_dict()
                quality_report['target']['imbalance_ratio'] = value_counts.min() / value_counts.max()
            else:  # Likely regression
                quality_report['target']['statistics'] = {
                    'mean': y.mean(),
                    'std': y.std(),
                    'min': y.min(),
                    'max': y.max()
                }
        
        return quality_report


class ModelUtils(LoggerMixin):
    """Utilities for model management and serialization."""
    
    @staticmethod
    def save_model(
        model: BaseEstimator,
        filepath: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None,
        compress: bool = True
    ) -> None:
        """Save model with metadata.
        
        Args:
            model: Trained model to save.
            filepath: Path to save the model.
            metadata: Additional metadata to save.
            compress: Whether to compress the model.
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare model data
        model_data = {
            'model': model,
            'metadata': metadata or {},
            'saved_at': datetime.now().isoformat(),
            'sklearn_version': None,  # Could import sklearn.__version__
        }
        
        # Add basic model info
        model_data['metadata'].update({
            'model_type': type(model).__name__,
            'model_params': model.get_params() if hasattr(model, 'get_params') else None
        })
        
        # Save using joblib for sklearn models
        if compress:
            joblib.dump(model_data, filepath, compress=3)
        else:
            joblib.dump(model_data, filepath)
        
        logger = ModelUtils().logger
        logger.info(f"Model saved to {filepath}")
    
    @staticmethod
    def load_model(filepath: Union[str, Path]) -> Tuple[BaseEstimator, Dict[str, Any]]:
        """Load model with metadata.
        
        Args:
            filepath: Path to the saved model.
            
        Returns:
            Tuple of (model, metadata).
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        # Load model data
        model_data = joblib.load(filepath)
        
        # Handle different save formats
        if isinstance(model_data, dict):
            model = model_data.get('model')
            metadata = model_data.get('metadata', {})
        else:
            # Backward compatibility - just the model
            model = model_data
            metadata = {}
        
        logger = ModelUtils().logger
        logger.info(f"Model loaded from {filepath}")
        
        return model, metadata
    
    @staticmethod
    def compare_model_sizes(models: Dict[str, BaseEstimator]) -> pd.DataFrame:
        """Compare memory usage of different models.
        
        Args:
            models: Dictionary of model names and instances.
            
        Returns:
            DataFrame with model size comparison.
        """
        import sys
        
        size_data = []
        
        for name, model in models.items():
            # Serialize to get approximate size
            serialized = pickle.dumps(model)
            size_bytes = len(serialized)
            
            # Get object size in memory
            memory_size = sys.getsizeof(model)
            
            size_data.append({
                'model_name': name,
                'serialized_size_kb': size_bytes / 1024,
                'memory_size_kb': memory_size / 1024,
                'model_type': type(model).__name__
            })
        
        return pd.DataFrame(size_data).sort_values('serialized_size_kb', ascending=False)


class ConfigUtils(LoggerMixin):
    """Configuration management utilities."""
    
    @staticmethod
    def save_config(config: Dict[str, Any], filepath: Union[str, Path]) -> None:
        """Save configuration to file.
        
        Args:
            config: Configuration dictionary.
            filepath: Path to save configuration.
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Determine format based on extension
        if filepath.suffix.lower() == '.json':
            with open(filepath, 'w') as f:
                json.dump(config, f, indent=2, default=str)
        elif filepath.suffix.lower() in ['.yml', '.yaml']:
            with open(filepath, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported config format: {filepath.suffix}")
        
        logger = ConfigUtils().logger
        logger.info(f"Configuration saved to {filepath}")
    
    @staticmethod
    def load_config(filepath: Union[str, Path]) -> Dict[str, Any]:
        """Load configuration from file.
        
        Args:
            filepath: Path to configuration file.
            
        Returns:
            Configuration dictionary.
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Config file not found: {filepath}")
        
        # Load based on extension
        if filepath.suffix.lower() == '.json':
            with open(filepath, 'r') as f:
                config = json.load(f)
        elif filepath.suffix.lower() in ['.yml', '.yaml']:
            with open(filepath, 'r') as f:
                config = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported config format: {filepath.suffix}")
        
        logger = ConfigUtils().logger
        logger.info(f"Configuration loaded from {filepath}")
        
        return config


class ExperimentTracker(LoggerMixin):
    """Simple experiment tracking utility."""
    
    def __init__(self, experiment_dir: Union[str, Path] = None):
        """Initialize experiment tracker.
        
        Args:
            experiment_dir: Directory to store experiment results.
        """
        self.experiment_dir = Path(experiment_dir or settings.REPORTS_DIR / "experiments")
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self.experiments = []
        
    def log_experiment(
        self,
        experiment_name: str,
        model_name: str,
        parameters: Dict[str, Any],
        metrics: Dict[str, float],
        notes: str = ""
    ) -> None:
        """Log experiment results.
        
        Args:
            experiment_name: Name of the experiment.
            model_name: Name of the model.
            parameters: Model parameters used.
            metrics: Performance metrics achieved.
            notes: Additional notes.
        """
        experiment_data = {
            'timestamp': datetime.now().isoformat(),
            'experiment_name': experiment_name,
            'model_name': model_name,
            'parameters': parameters,
            'metrics': metrics,
            'notes': notes
        }
        
        self.experiments.append(experiment_data)
        
        # Save to file
        experiment_file = self.experiment_dir / f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        ConfigUtils.save_config(experiment_data, experiment_file)
        
        self.logger.info(f"Experiment logged: {experiment_name}")
    
    def get_experiment_summary(self) -> pd.DataFrame:
        """Get summary of all experiments.
        
        Returns:
            DataFrame with experiment results.
        """
        if not self.experiments:
            return pd.DataFrame()
        
        # Flatten experiment data
        summary_data = []
        for exp in self.experiments:
            row = {
                'timestamp': exp['timestamp'],
                'experiment_name': exp['experiment_name'],
                'model_name': exp['model_name'],
                'notes': exp['notes']
            }
            
            # Add parameters
            for key, value in exp['parameters'].items():
                row[f'param_{key}'] = value
            
            # Add metrics
            for key, value in exp['metrics'].items():
                row[f'metric_{key}'] = value
            
            summary_data.append(row)
        
        return pd.DataFrame(summary_data)
    
    def load_experiments_from_dir(self) -> None:
        """Load all experiments from the experiment directory."""
        self.experiments = []
        
        for experiment_file in self.experiment_dir.glob("*.json"):
            try:
                experiment_data = ConfigUtils.load_config(experiment_file)
                self.experiments.append(experiment_data)
            except Exception as e:
                self.logger.warning(f"Could not load experiment {experiment_file}: {e}")
        
        self.logger.info(f"Loaded {len(self.experiments)} experiments")


class PerformanceProfiler(LoggerMixin):
    """Performance profiling utilities."""
    
    @staticmethod
    def profile_pipeline_steps(
        pipeline: BaseEstimator,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        n_runs: int = 3
    ) -> Dict[str, Dict[str, float]]:
        """Profile performance of individual pipeline steps.
        
        Args:
            pipeline: Pipeline to profile.
            X: Feature matrix.
            y: Target vector (for fitting).
            n_runs: Number of runs for averaging.
            
        Returns:
            Dictionary with timing information for each step.
        """
        import time
        
        logger = PerformanceProfiler().logger
        results = {}
        
        # Check if it's a pipeline
        if not hasattr(pipeline, 'steps'):
            logger.warning("Not a pipeline object, profiling entire model")
            return PerformanceProfiler._profile_single_estimator(pipeline, X, y, n_runs)
        
        # Profile each step
        for step_name, estimator in pipeline.steps:
            step_times = {'fit': [], 'transform': []}
            
            for run in range(n_runs):
                # Fit timing
                if hasattr(estimator, 'fit'):
                    start_time = time.time()
                    try:
                        if hasattr(estimator, 'fit_transform'):
                            estimator.fit(X, y)
                        else:
                            estimator.fit(X)
                    except Exception as e:
                        logger.warning(f"Could not fit {step_name}: {e}")
                        continue
                    
                    fit_time = time.time() - start_time
                    step_times['fit'].append(fit_time)
                
                # Transform timing
                if hasattr(estimator, 'transform'):
                    start_time = time.time()
                    try:
                        X_transformed = estimator.transform(X)
                        transform_time = time.time() - start_time
                        step_times['transform'].append(transform_time)
                        X = X_transformed  # Use transformed data for next step
                    except Exception as e:
                        logger.warning(f"Could not transform {step_name}: {e}")
                        continue
            
            # Calculate averages
            results[step_name] = {
                'avg_fit_time': np.mean(step_times['fit']) if step_times['fit'] else 0,
                'avg_transform_time': np.mean(step_times['transform']) if step_times['transform'] else 0,
                'std_fit_time': np.std(step_times['fit']) if step_times['fit'] else 0,
                'std_transform_time': np.std(step_times['transform']) if step_times['transform'] else 0
            }
        
        return results
    
    @staticmethod
    def _profile_single_estimator(
        estimator: BaseEstimator,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        n_runs: int = 3
    ) -> Dict[str, Dict[str, float]]:
        """Profile a single estimator."""
        import time
        
        fit_times = []
        predict_times = []
        
        for run in range(n_runs):
            # Fit timing
            start_time = time.time()
            estimator.fit(X, y)
            fit_time = time.time() - start_time
            fit_times.append(fit_time)
            
            # Predict timing
            start_time = time.time()
            estimator.predict(X[:100])  # Use subset for prediction timing
            predict_time = time.time() - start_time
            predict_times.append(predict_time)
        
        return {
            'estimator': {
                'avg_fit_time': np.mean(fit_times),
                'avg_predict_time': np.mean(predict_times),
                'std_fit_time': np.std(fit_times),
                'std_predict_time': np.std(predict_times)
            }
        }
    

class VisualizationUtils(LoggerMixin):
    """Utilities for saving and managing visualizations."""
    
    @staticmethod
    def save_figure(
        fig,
        filename: str,
        folder: Union[str, Path] = None,
        subfolder: str = None,
        dpi: int = 300,
        bbox_inches: str = "tight",
        add_timestamp: bool = True,
        formats: List[str] = None
    ) -> Dict[str, str]:
        """Save matplotlib figure to results folder with proper naming.
        
        Args:
            fig: Matplotlib figure object.
            filename: Base filename (without extension).
            folder: Base folder to save to. Defaults to results/figures.
            subfolder: Subfolder within the base folder.
            dpi: Resolution for saving.
            bbox_inches: Bounding box setting.
            add_timestamp: Whether to add timestamp to filename.
            formats: List of formats to save ['png', 'pdf', 'svg']. Defaults to ['png'].
            
        Returns:
            Dictionary with format -> filepath mappings.
        """
        import matplotlib.pyplot as plt
        
        # Set defaults
        if folder is None:
            folder = settings.FIGURES_DIR if hasattr(settings, 'FIGURES_DIR') else Path("results/figures")
        if formats is None:
            formats = ['png']
        
        folder = Path(folder)
        
        # Add subfolder if specified
        if subfolder:
            folder = folder / subfolder
        
        # Create directory
        folder.mkdir(parents=True, exist_ok=True)
        
        # Prepare filename
        if add_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_filename = f"{timestamp}_{filename}"
        else:
            base_filename = filename
        
        # Save in multiple formats
        saved_paths = {}
        logger = VisualizationUtils().logger
        
        for fmt in formats:
            filepath = folder / f"{base_filename}.{fmt}"
            
            try:
                fig.savefig(filepath, dpi=dpi, bbox_inches=bbox_inches, format=fmt)
                saved_paths[fmt] = str(filepath)
                logger.info(f"📁 Figure saved: {filepath}")
            except Exception as e:
                logger.error(f"Failed to save figure in {fmt} format: {e}")
        
        return saved_paths
    
    @staticmethod
    def create_results_summary(
        results_dir: Union[str, Path] = None,
        output_file: str = "visualization_summary.html"
    ) -> str:
        """Create HTML summary of all saved visualizations.
        
        Args:
            results_dir: Directory to scan for figures.
            output_file: Name of the HTML summary file.
            
        Returns:
            Path to the created HTML file.
        """
        if results_dir is None:
            results_dir = settings.FIGURES_DIR if hasattr(settings, 'FIGURES_DIR') else Path("results/figures")
        
        results_dir = Path(results_dir)
        
        # Find all image files
        image_extensions = ['.png', '.jpg', '.jpeg', '.pdf', '.svg']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(results_dir.rglob(f"*{ext}"))
        
        # Group by subfolder
        grouped_images = {}
        for img_path in image_files:
            relative_path = img_path.relative_to(results_dir)
            subfolder = str(relative_path.parent) if relative_path.parent != Path('.') else 'root'
            
            if subfolder not in grouped_images:
                grouped_images[subfolder] = []
            
            grouped_images[subfolder].append({
                'name': img_path.name,
                'path': str(relative_path),
                'size': img_path.stat().st_size,
                'modified': datetime.fromtimestamp(img_path.stat().st_mtime)
            })
        
        # Generate HTML
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Visualization Summary - sklearn-mastery</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; }}
                .subfolder {{ background-color: #f9f9f9; padding: 15px; margin: 10px 0; border-radius: 5px; }}
                .image-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 20px; }}
                .image-item {{ border: 1px solid #ddd; padding: 10px; border-radius: 5px; }}
                .image-item img {{ max-width: 100%; height: auto; }}
                .image-info {{ font-size: 12px; color: #666; margin-top: 10px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>📊 Visualization Summary</h1>
                <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>Total images: {len(image_files)}</p>
            </div>
        """
        
        for subfolder, images in grouped_images.items():
            html_content += f"""
            <div class="section">
                <div class="subfolder">
                    <h2>📁 {subfolder.replace('_', ' ').title()}</h2>
                    <p>{len(images)} images</p>
                    <div class="image-grid">
            """
            
            for img in sorted(images, key=lambda x: x['modified'], reverse=True):
                # Only show images that can be displayed in HTML
                if img['path'].lower().endswith(('.png', '.jpg', '.jpeg')):
                    html_content += f"""
                        <div class="image-item">
                            <img src="{img['path']}" alt="{img['name']}">
                            <div class="image-info">
                                <strong>{img['name']}</strong><br>
                                Size: {img['size'] // 1024} KB<br>
                                Modified: {img['modified'].strftime('%Y-%m-%d %H:%M')}
                            </div>
                        </div>
                    """
                else:
                    html_content += f"""
                        <div class="image-item">
                            <p><strong>{img['name']}</strong></p>
                            <p>📄 {img['path'].split('.')[-1].upper()} file</p>
                            <div class="image-info">
                                Size: {img['size'] // 1024} KB<br>
                                Modified: {img['modified'].strftime('%Y-%m-%d %H:%M')}
                            </div>
                        </div>
                    """
            
            html_content += """
                    </div>
                </div>
            </div>
            """
        
        html_content += """
        </body>
        </html>
        """
        
        # Save HTML file
        html_path = results_dir / output_file
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger = VisualizationUtils().logger
        logger.info(f"📋 Visualization summary created: {html_path}")
        
        return str(html_path)
    
    @staticmethod
    def setup_results_directories() -> Dict[str, Path]:
        """Set up the standard results directory structure.
        
        Returns:
            Dictionary mapping directory purposes to paths.
        """
        base_dir = Path("results")
        
        directories = {
            'base': base_dir,
            'figures': base_dir / 'figures',
            'models': base_dir / 'models',
            'reports': base_dir / 'reports',
            'data_generation': base_dir / 'figures' / 'data_generation',
            'preprocessing': base_dir / 'figures' / 'preprocessing',
            'classification': base_dir / 'figures' / 'classification',
            'regression': base_dir / 'figures' / 'regression',
            'clustering': base_dir / 'figures' / 'clustering',
            'model_comparison': base_dir / 'figures' / 'model_comparison',
            'hyperparameter_optimization': base_dir / 'figures' / 'hyperparameter_optimization',
            'interpretability': base_dir / 'figures' / 'interpretability'
        }
        
        # Create all directories
        for purpose, path in directories.items():
            path.mkdir(parents=True, exist_ok=True)
        
        logger = VisualizationUtils().logger
        logger.info(f"📁 Results directory structure created: {base_dir}")
        
        return directories


# Convenience function for easy import
def save_figure(fig, filename: str, subfolder: str = None, **kwargs) -> Dict[str, str]:
    """Convenience function for saving figures.
    
    Args:
        fig: Matplotlib figure object.
        filename: Base filename.
        subfolder: Subfolder within results/figures.
        **kwargs: Additional arguments for VisualizationUtils.save_figure.
        
    Returns:
        Dictionary with format -> filepath mappings.
    """
    return VisualizationUtils.save_figure(fig, filename, subfolder=subfolder, **kwargs)