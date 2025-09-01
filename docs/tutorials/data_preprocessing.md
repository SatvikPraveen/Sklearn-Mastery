# Data Preprocessing Tutorial

Data preprocessing is a crucial step in any machine learning pipeline. This tutorial covers comprehensive preprocessing techniques using the Scikit-Learn Mastery Project's custom transformers and utilities.

## Table of Contents

1. [Overview](#overview)
2. [Data Loading and Exploration](#data-loading-and-exploration)
3. [Handling Missing Values](#handling-missing-values)
4. [Outlier Detection and Removal](#outlier-detection-and-removal)
5. [Data Scaling and Normalization](#data-scaling-and-normalization)
6. [Categorical Data Encoding](#categorical-data-encoding)
7. [Feature Selection](#feature-selection)
8. [Data Transformation](#data-transformation)
9. [Building Preprocessing Pipelines](#building-preprocessing-pipelines)
10. [Advanced Preprocessing Techniques](#advanced-preprocessing-techniques)
11. [Best Practices](#best-practices)

## Overview

Data preprocessing transforms raw data into a format suitable for machine learning algorithms. The project provides custom transformers that integrate seamlessly with sklearn pipelines while offering enhanced functionality.

### Key Preprocessing Steps

1. **Data Quality Assessment**: Identify issues in the data
2. **Missing Value Handling**: Impute or remove missing values
3. **Outlier Treatment**: Detect and handle outliers
4. **Scaling/Normalization**: Standardize feature scales
5. **Encoding**: Convert categorical variables
6. **Feature Selection**: Choose relevant features
7. **Transformation**: Create new features or transform existing ones

## Data Loading and Exploration

Let's start by creating and exploring a realistic dataset with common data quality issues:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Project imports
from src.data.generators import ClassificationDataGenerator
from src.preprocessing.preprocessor import DataPreprocessor
from src.preprocessing.validator import DataValidator
from src.utils.visualization import DataVisualizer

# Set style for better plots
plt.style.use('seaborn-v0_8')
np.random.seed(42)
```

### Create Realistic Dataset with Issues

```python
# Generate base dataset
generator = ClassificationDataGenerator()
X_clean, y = generator.generate_basic_classification(
    n_samples=1000,
    n_features=8,
    n_informative=6,
    n_redundant=1,
    n_classes=3,
    random_state=42
)

# Create DataFrame with meaningful column names
feature_names = [
    'age', 'income', 'education_years', 'experience_years',
    'credit_score', 'debt_ratio', 'savings', 'expenses'
]
df = pd.DataFrame(X_clean, columns=feature_names)
df['target'] = y

# Introduce realistic data issues
np.random.seed(42)

# 1. Missing values (10% random missing)
missing_mask = np.random.random(df.shape) < 0.1
df = df.mask(missing_mask)

# 2. Outliers in specific columns
outlier_indices = np.random.choice(df.index, size=50, replace=False)
df.loc[outlier_indices, 'income'] *= 10  # Extreme income values
df.loc[outlier_indices[:25], 'debt_ratio'] = np.random.uniform(5, 20, 25)  # Extreme debt ratios

# 3. Categorical data
categories = ['low', 'medium', 'high']
df['income_category'] = pd.cut(df['income'], bins=3, labels=categories)

# 4. Mixed data types
df['has_savings'] = (df['savings'] > df['savings'].median()).astype('category')

print("Dataset shape:", df.shape)
print("\nDataset info:")
print(df.info())
```

### Data Quality Assessment

```python
# Use project's data validator
validator = DataValidator()

# Comprehensive data quality report
quality_report = validator.validate_data(df.drop('target', axis=1), df['target'])

print("Data Quality Report:")
print("-" * 50)
print(f"Missing values: {quality_report['missing_values']}")
print(f"Outliers detected: {quality_report['outliers']}")
print(f"Data types: {quality_report['dtypes']}")
print(f"Class balance: {quality_report['class_balance']}")
```

### Visual Data Exploration

```python
# Create visualizer
data_viz = DataVisualizer()

# Plot missing value pattern
def plot_missing_values(df):
    """Visualize missing value patterns."""
    missing_data = df.isnull()

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Missing value heatmap
    sns.heatmap(missing_data, yticklabels=False, cbar=True, cmap='viridis', ax=axes[0,0])
    axes[0,0].set_title('Missing Value Pattern')

    # Missing value counts
    missing_counts = missing_data.sum()
    missing_counts[missing_counts > 0].plot(kind='bar', ax=axes[0,1])
    axes[0,1].set_title('Missing Values by Column')
    axes[0,1].tick_params(axis='x', rotation=45)

    # Missing value percentage
    missing_pct = (missing_data.sum() / len(df)) * 100
    missing_pct[missing_pct > 0].plot(kind='bar', ax=axes[1,0], color='orange')
    axes[1,0].set_title('Missing Value Percentage')
    axes[1,0].tick_params(axis='x', rotation=45)

    # Data completeness
    completeness = ((1 - missing_data.sum() / len(df)) * 100)
    completeness.plot(kind='bar', ax=axes[1,1], color='green')
    axes[1,1].set_title('Data Completeness (%)')
    axes[1,1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()

plot_missing_values(df)
```

## Handling Missing Values

The project provides flexible missing value imputation strategies:

### Simple Imputation

```python
from src.pipelines.custom_transformers import MissingValueImputer

# Separate numerical and categorical columns
numerical_cols = df.select_dtypes(include=[np.number]).columns.drop('target')
categorical_cols = df.select_dtypes(include=['object', 'category']).columns

print("Numerical columns:", list(numerical_cols))
print("Categorical columns:", list(categorical_cols))

# Mean imputation for numerical features
mean_imputer = MissingValueImputer(strategy='mean')
X_numerical = df[numerical_cols]
X_numerical_imputed = mean_imputer.fit_transform(X_numerical)

print("Missing values before imputation:", X_numerical.isnull().sum().sum())
print("Missing values after imputation:", pd.DataFrame(X_numerical_imputed, columns=numerical_cols).isnull().sum().sum())
```

### Advanced Imputation Strategies

```python
# Different strategies for different column types
imputation_strategies = {
    'mean': ['age', 'income', 'credit_score'],
    'median': ['education_years', 'experience_years', 'debt_ratio'],
    'most_frequent': ['income_category', 'has_savings'],
    'constant': ['savings', 'expenses']  # Fill with 0
}

# Apply different strategies
X_imputed = df.copy()

for strategy, columns in imputation_strategies.items():
    valid_columns = [col for col in columns if col in df.columns]
    if valid_columns:
        if strategy == 'constant':
            imputer = MissingValueImputer(strategy=strategy, fill_value=0)
        else:
            imputer = MissingValueImputer(strategy=strategy)

        X_imputed[valid_columns] = imputer.fit_transform(X_imputed[valid_columns])

print("Missing values after advanced imputation:")
print(X_imputed.isnull().sum())
```

### Imputation Quality Assessment

```python
def assess_imputation_quality(original, imputed, columns):
    """Assess the quality of imputation."""
    results = {}

    for col in columns:
        if col in original.columns and col in imputed.columns:
            # Original non-missing values
            orig_mask = ~original[col].isnull()
            orig_values = original.loc[orig_mask, col]

            # Imputed values for originally missing positions
            missing_mask = original[col].isnull()
            imputed_values = imputed.loc[missing_mask, col]

            if len(orig_values) > 0 and len(imputed_values) > 0:
                results[col] = {
                    'original_mean': orig_values.mean() if pd.api.types.is_numeric_dtype(orig_values) else None,
                    'imputed_mean': imputed_values.mean() if pd.api.types.is_numeric_dtype(imputed_values) else None,
                    'original_missing_pct': (missing_mask.sum() / len(original)) * 100
                }

    return results

# Assess imputation quality
quality_assessment = assess_imputation_quality(df, X_imputed, numerical_cols)

print("Imputation Quality Assessment:")
for col, metrics in quality_assessment.items():
    if metrics['original_mean'] is not None:
        print(f"{col}:")
        print(f"  Original mean: {metrics['original_mean']:.2f}")
        print(f"  Imputed mean: {metrics['imputed_mean']:.2f}")
        print(f"  Missing percentage: {metrics['original_missing_pct']:.1f}%")
```

## Outlier Detection and Removal

Outliers can significantly impact model performance. The project provides multiple outlier detection methods:

### IQR Method

```python
from src.pipelines.custom_transformers import OutlierRemover

# IQR-based outlier removal
iqr_remover = OutlierRemover(method='iqr', threshold=1.5)

# Apply to numerical data
X_no_outliers_iqr = iqr_remover.fit_transform(X_imputed[numerical_cols])
removed_indices_iqr = iqr_remover.get_outlier_indices()

print(f"IQR method removed {len(removed_indices_iqr)} outliers")
print(f"Data shape before: {X_imputed.shape}")
print(f"Data shape after: {X_no_outliers_iqr.shape}")
```

### Z-Score Method

```python
# Z-score based outlier removal
zscore_remover = OutlierRemover(method='zscore', threshold=3.0)
X_no_outliers_z = zscore_remover.fit_transform(X_imputed[numerical_cols])
removed_indices_z = zscore_remover.get_outlier_indices()

print(f"Z-score method removed {len(removed_indices_z)} outliers")
```

### Isolation Forest Method

```python
# Isolation Forest for outlier detection
isolation_remover = OutlierRemover(method='isolation_forest', contamination=0.05)
X_no_outliers_iso = isolation_remover.fit_transform(X_imputed[numerical_cols])
removed_indices_iso = isolation_remover.get_outlier_indices()

print(f"Isolation Forest removed {len(removed_indices_iso)} outliers")
```

### Outlier Visualization

```python
def visualize_outliers(original_data, cleaned_data, feature_name):
    """Visualize outlier removal effects."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Box plots
    axes[0].boxplot([original_data[feature_name].dropna(),
                     cleaned_data[feature_name] if feature_name in cleaned_data.columns
                     else cleaned_data.iloc[:, 0]],
                   labels=['Original', 'Cleaned'])
    axes[0].set_title(f'{feature_name} - Box Plot Comparison')

    # Histograms
    axes[1].hist(original_data[feature_name].dropna(), alpha=0.7, label='Original', bins=30)
    if feature_name in cleaned_data.columns:
        axes[1].hist(cleaned_data[feature_name], alpha=0.7, label='Cleaned', bins=30)
    else:
        axes[1].hist(cleaned_data.iloc[:, 0], alpha=0.7, label='Cleaned', bins=30)
    axes[1].set_title(f'{feature_name} - Distribution Comparison')
    axes[1].legend()

    # Q-Q plot
    from scipy import stats
    stats.probplot(original_data[feature_name].dropna(), dist="norm", plot=axes[2])
    axes[2].set_title(f'{feature_name} - Q-Q Plot (Original)')

    plt.tight_layout()
    plt.show()

# Visualize outlier removal for income
visualize_outliers(X_imputed, pd.DataFrame(X_no_outliers_iqr, columns=numerical_cols), 'income')
```

## Data Scaling and Normalization

Feature scaling ensures all features contribute equally to model training:

### Standard Scaling (Z-score normalization)

```python
from src.pipelines.custom_transformers import ScalerTransformer

# Standard scaling
standard_scaler = ScalerTransformer(method='standard')
X_scaled_standard = standard_scaler.fit_transform(X_no_outliers_iqr)

print("Standard Scaling Results:")
print(f"Mean: {np.mean(X_scaled_standard, axis=0)}")
print(f"Std: {np.std(X_scaled_standard, axis=0)}")
```

### Min-Max Scaling

```python
# Min-Max scaling
minmax_scaler = ScalerTransformer(method='minmax')
X_scaled_minmax = minmax_scaler.fit_transform(X_no_outliers_iqr)

print("\nMin-Max Scaling Results:")
print(f"Min: {np.min(X_scaled_minmax, axis=0)}")
print(f"Max: {np.max(X_scaled_minmax, axis=0)}")
```

### Robust Scaling

```python
# Robust scaling (less sensitive to outliers)
robust_scaler = ScalerTransformer(method='robust')
X_scaled_robust = robust_scaler.fit_transform(X_no_outliers_iqr)

print("\nRobust Scaling Results:")
print(f"Median: {np.median(X_scaled_robust, axis=0)}")
print(f"IQR: {np.percentile(X_scaled_robust, 75, axis=0) - np.percentile(X_scaled_robust, 25, axis=0)}")
```

### Scaling Comparison

```python
def compare_scaling_methods(original_data, feature_idx=0):
    """Compare different scaling methods."""
    scalers = {
        'Original': lambda x: x,
        'Standard': lambda x: ScalerTransformer(method='standard').fit_transform(x),
        'MinMax': lambda x: ScalerTransformer(method='minmax').fit_transform(x),
        'Robust': lambda x: ScalerTransformer(method='robust').fit_transform(x)
    }

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.ravel()

    for idx, (name, scaler_func) in enumerate(scalers.items()):
        if name == 'Original':
            data = original_data[:, feature_idx]
        else:
            scaled_data = scaler_func(original_data)
            data = scaled_data[:, feature_idx]

        axes[idx].hist(data, bins=30, alpha=0.7)
        axes[idx].set_title(f'{name} Scaling')
        axes[idx].set_xlabel('Value')
        axes[idx].set_ylabel('Frequency')

        # Add statistics
        mean_val = np.mean(data)
        std_val = np.std(data)
        axes[idx].axvline(mean_val, color='red', linestyle='--',
                         label=f'Mean: {mean_val:.2f}')
        axes[idx].axvline(mean_val + std_val, color='orange', linestyle='--',
                         label=f'Mean + Std: {mean_val + std_val:.2f}')
        axes[idx].legend()

    plt.tight_layout()
    plt.show()

compare_scaling_methods(X_no_outliers_iqr)
```

## Categorical Data Encoding

Converting categorical variables to numerical format is essential for most ML algorithms:

### One-Hot Encoding

```python
from src.pipelines.custom_transformers import CategoryEncoder

# One-hot encoding for categorical features
onehot_encoder = CategoryEncoder(method='onehot', columns=['income_category'])

# Prepare data with categorical columns
categorical_df = X_imputed[['income_category', 'has_savings']].copy()
categorical_df = categorical_df.dropna()  # Remove NaN for encoding

# Apply one-hot encoding
encoded_onehot = onehot_encoder.fit_transform(categorical_df[['income_category']])

print("One-Hot Encoding Results:")
print(f"Original shape: {categorical_df[['income_category']].shape}")
print(f"Encoded shape: {encoded_onehot.shape}")
print(f"New columns: {encoded_onehot.columns.tolist()}")
```

### Label Encoding

```python
# Label encoding for ordinal categorical data
label_encoder = CategoryEncoder(method='label', columns=['income_category'])
encoded_label = label_encoder.fit_transform(categorical_df[['income_category']])

print("\nLabel Encoding Results:")
print("Original values:", categorical_df['income_category'].unique())
print("Encoded values:", encoded_label['income_category'].unique())
```

### Target Encoding

```python
# Target encoding (mean encoding)
target_encoder = CategoryEncoder(method='target', columns=['income_category'])

# Need target variable for target encoding
y_categorical = X_imputed['target'].dropna()
categorical_with_target = categorical_df.loc[y_categorical.index]

encoded_target = target_encoder.fit_transform(
    categorical_with_target[['income_category']],
    y_categorical
)

print("\nTarget Encoding Results:")
print("Category means:")
for category in categorical_df['income_category'].unique():
    if pd.notna(category):
        mask = categorical_with_target['income_category'] == category
        if mask.sum() > 0:
            target_mean = y_categorical[mask].mean()
            encoded_mean = encoded_target.loc[mask, 'income_category'].iloc[0]
            print(f"  {category}: {target_mean:.3f} -> {encoded_mean:.3f}")
```

### Encoding Comparison

```python
def compare_encoding_methods(data, target, categorical_col):
    """Compare different encoding methods."""
    encoders = {
        'One-Hot': CategoryEncoder(method='onehot', columns=[categorical_col]),
        'Label': CategoryEncoder(method='label', columns=[categorical_col]),
        'Target': CategoryEncoder(method='target', columns=[categorical_col])
    }

    results = {}

    for name, encoder in encoders.items():
        if name == 'Target':
            encoded = encoder.fit_transform(data[[categorical_col]], target)
        else:
            encoded = encoder.fit_transform(data[[categorical_col]])

        results[name] = {
            'shape': encoded.shape,
            'columns': encoded.columns.tolist() if hasattr(encoded, 'columns') else ['encoded'],
            'sample': encoded.head()
        }

    return results

# Compare encoding methods
encoding_comparison = compare_encoding_methods(
    categorical_with_target, y_categorical, 'income_category'
)

for method, result in encoding_comparison.items():
    print(f"\n{method} Encoding:")
    print(f"  Shape: {result['shape']}")
    print(f"  Columns: {result['columns']}")
```

## Feature Selection

Selecting relevant features improves model performance and reduces overfitting:

### Variance Threshold Selection

```python
from src.pipelines.custom_transformers import VarianceThresholdSelector

# Remove low-variance features
variance_selector = VarianceThresholdSelector(threshold=0.01)
X_selected_variance = variance_selector.fit_transform(X_scaled_standard)

print("Variance Threshold Selection:")
print(f"Original features: {X_scaled_standard.shape[1]}")
print(f"Selected features: {X_selected_variance.shape[1]}")
print(f"Removed features: {X_scaled_standard.shape[1] - X_selected_variance.shape[1]}")

# Get feature support
support_mask = variance_selector.get_support()
selected_features = np.array(numerical_cols)[support_mask]
print(f"Selected features: {selected_features.tolist()}")
```

### Correlation-Based Selection

```python
from src.pipelines.custom_transformers import CorrelationSelector

# Remove highly correlated features
correlation_selector = CorrelationSelector(threshold=0.9)
X_selected_corr = correlation_selector.fit_transform(X_scaled_standard)

print("\nCorrelation-Based Selection:")
print(f"Original features: {X_scaled_standard.shape[1]}")
print(f"Selected features: {X_selected_corr.shape[1]}")

# Visualize correlation matrix
correlation_matrix = correlation_selector.get_correlation_matrix()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix,
            annot=True,
            cmap='coolwarm',
            center=0,
            xticklabels=numerical_cols,
            yticklabels=numerical_cols)
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.show()
```

### Univariate Feature Selection

```python
from src.pipelines.custom_transformers import FeatureSelector

# Univariate feature selection based on statistical tests
univariate_selector = FeatureSelector(method='univariate', k=5)

# Need to align target with cleaned data
y_aligned = X_imputed['target'].iloc[:X_scaled_standard.shape[0]]
X_selected_univariate = univariate_selector.fit_transform(X_scaled_standard, y_aligned)

print("\nUnivariate Feature Selection:")
print(f"Selected top {univariate_selector.k} features")
print(f"Original features: {X_scaled_standard.shape[1]}")
print(f"Selected features: {X_selected_univariate.shape[1]}")

# Get selected feature indices
selected_indices = univariate_selector.get_selected_features()
selected_feature_names = np.array(numerical_cols)[selected_indices]
print(f"Selected features: {selected_feature_names.tolist()}")
```

### Recursive Feature Elimination

```python
# RFE with Random Forest
rfe_selector = FeatureSelector(method='rfe', k=4)
X_selected_rfe = rfe_selector.fit_transform(X_scaled_standard, y_aligned)

print("\nRecursive Feature Elimination:")
print(f"Selected {rfe_selector.k} features using RFE")
print(f"Original features: {X_scaled_standard.shape[1]}")
print(f"Selected features: {X_selected_rfe.shape[1]}")

# Get feature rankings
selected_indices_rfe = rfe_selector.get_selected_features()
selected_feature_names_rfe = np.array(numerical_cols)[selected_indices_rfe]
print(f"Selected features: {selected_feature_names_rfe.tolist()}")
```

### Feature Selection Comparison

```python
def compare_feature_selection_methods(X, y, feature_names):
    """Compare different feature selection methods."""
    selectors = {
        'Variance (0.01)': VarianceThresholdSelector(threshold=0.01),
        'Correlation (0.9)': CorrelationSelector(threshold=0.9),
        'Univariate (k=5)': FeatureSelector(method='univariate', k=5),
        'RFE (k=5)': FeatureSelector(method='rfe', k=5)
    }

    results = {}

    for name, selector in selectors.items():
        if 'Univariate' in name or 'RFE' in name:
            X_selected = selector.fit_transform(X, y)
            if hasattr(selector, 'get_selected_features'):
                selected_indices = selector.get_selected_features()
            else:
                selected_indices = np.where(selector.get_support())[0]
        else:
            X_selected = selector.fit_transform(X)
            selected_indices = np.where(selector.get_support())[0]

        selected_features = feature_names[selected_indices]

        results[name] = {
            'n_features': len(selected_features),
            'features': selected_features.tolist(),
            'shape': X_selected.shape
        }

    return results

# Compare feature selection methods
selection_comparison = compare_feature_selection_methods(
    X_scaled_standard, y_aligned, np.array(numerical_cols)
)

print("Feature Selection Comparison:")
print("-" * 60)
for method, result in selection_comparison.items():
    print(f"{method}:")
    print(f"  Features selected: {result['n_features']}")
    print(f"  Features: {result['features']}")
    print(f"  Shape: {result['shape']}")
    print()
```

## Data Transformation

Transform features to improve model performance:

### Polynomial Features

```python
from src.pipelines.custom_transformers import PolynomialFeatureGenerator

# Generate polynomial features
poly_generator = PolynomialFeatureGenerator(degree=2, include_bias=False)
X_poly = poly_generator.fit_transform(X_selected_univariate)

print("Polynomial Feature Generation:")
print(f"Original features: {X_selected_univariate.shape[1]}")
print(f"Polynomial features: {X_poly.shape[1]}")
print(f"Added features: {X_poly.shape[1] - X_selected_univariate.shape[1]}")
```

### Interaction Features

```python
from src.pipelines.custom_transformers import InteractionFeatureGenerator

# Generate interaction features
interaction_generator = InteractionFeatureGenerator(degree=2)
X_interactions = interaction_generator.fit_transform(X_selected_univariate)

print("\nInteraction Feature Generation:")
print(f"Original features: {X_selected_univariate.shape[1]}")
print(f"With interactions: {X_interactions.shape[1]}")
print(f"Interaction features: {X_interactions.shape[1] - X_selected_univariate.shape[1]}")
```

### Binning/Discretization

```python
from src.pipelines.custom_transformers import BinningTransformer

# Bin continuous features
binning_transformer = BinningTransformer(n_bins=5, strategy='uniform', encode='ordinal')
X_binned = binning_transformer.fit_transform(X_selected_univariate)

print("\nFeature Binning:")
print(f"Shape: {X_binned.shape}")
print(f"Bin ranges for first feature:")

# Show bin edges for first feature
bin_edges = binning_transformer.model.bin_edges_[0]
for i in range(len(bin_edges) - 1):
    print(f"  Bin {i}: [{bin_edges[i]:.2f}, {bin_edges[i+1]:.2f})")
```

### Log Transformation

```python
from src.pipelines.custom_transformers import LogTransformer

# Apply log transformation (for positive features)
# Make data positive by shifting
X_positive = X_selected_univariate - X_selected_univariate.min(axis=0) + 1

log_transformer = LogTransformer(method='log1p')
X_log = log_transformer.fit_transform(X_positive)

print("\nLog Transformation:")
print(f"Shape: {X_log.shape}")
print(f"Original range: [{X_positive.min():.2f}, {X_positive.max():.2f}]")
print(f"Log-transformed range: [{X_log.min():.2f}, {X_log.max():.2f}]")

# Visualize transformation effect
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(X_positive[:, 0], bins=30, alpha=0.7, label='Original')
plt.title('Original Distribution')
plt.xlabel('Value')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
plt.hist(X_log[:, 0], bins=30, alpha=0.7, label='Log-transformed', color='orange')
plt.title('Log-transformed Distribution')
plt.xlabel('Value')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()
```

## Building Preprocessing Pipelines

Combine multiple preprocessing steps into coherent pipelines:

### Basic Preprocessing Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Define preprocessing steps for different data types
numerical_pipeline = Pipeline([
    ('imputation', MissingValueImputer(strategy='median')),
    ('outlier_removal', OutlierRemover(method='iqr')),
    ('scaling', ScalerTransformer(method='standard'))
])

categorical_pipeline = Pipeline([
    ('imputation', MissingValueImputer(strategy='most_frequent')),
    ('encoding', CategoryEncoder(method='onehot'))
])

# Combine pipelines
preprocessor = ColumnTransformer([
    ('numerical', numerical_pipeline, numerical_cols),
    ('categorical', categorical_pipeline, ['income_category'])
])

# Apply preprocessing
X_preprocessed = preprocessor.fit_transform(X_imputed.drop('target', axis=1))

print("Combined Preprocessing Pipeline:")
print(f"Input shape: {X_imputed.drop('target', axis=1).shape}")
print(f"Output shape: {X_preprocessed.shape}")
```

### Advanced Preprocessing Pipeline

```python
# Advanced pipeline with feature engineering
advanced_pipeline = Pipeline([
    # Step 1: Basic cleaning
    ('imputation', MissingValueImputer(strategy='mean')),
    ('outlier_removal', OutlierRemover(method='iqr')),

    # Step 2: Feature engineering
    ('polynomial', PolynomialFeatureGenerator(degree=2, include_bias=False)),

    # Step 3: Feature selection
    ('variance_selection', VarianceThresholdSelector(threshold=0.01)),
    ('correlation_selection', CorrelationSelector(threshold=0.95)),
    ('feature_selection', FeatureSelector(method='univariate', k=15)),

    # Step 4: Final scaling
    ('scaling', ScalerTransformer(method='robust'))
])

# Apply advanced pipeline
X_advanced = advanced_pipeline.fit_transform(X_imputed[numerical_cols], y_aligned)

print("\nAdvanced Preprocessing Pipeline:")
print(f"Input shape: {X_imputed[numerical_cols].shape}")
print(f"Output shape: {X_advanced.shape}")
print(f"Transformation ratio: {X_advanced.shape[1] / X_imputed[numerical_cols].shape[1]:.2f}")
```

### Pipeline Visualization

```python
def visualize_pipeline_effects(original, intermediate_steps, step_names):
    """Visualize the effects of pipeline steps."""
    n_steps = len(intermediate_steps)
    fig, axes = plt.subplots(2, (n_steps + 1) // 2, figsize=(15, 8))
    axes = axes.ravel() if n_steps > 1 else [axes]

    # Plot original data
    axes[0].hist(original[:, 0], bins=30, alpha=0.7)
    axes[0].set_title('Original Data')
    axes[0].set_xlabel('Value')
    axes[0].set_ylabel('Frequency')

    # Plot each transformation step
    for i, (data, step_name) in enumerate(zip(intermediate_steps, step_names)):
        if i + 1 < len(axes):
            axes[i + 1].hist(data[:, 0], bins=30, alpha=0.7)
            axes[i + 1].set_title(f'After {step_name}')
            axes[i + 1].set_xlabel('Value')
            axes[i + 1].set_ylabel('Frequency')

    # Hide empty subplots
    for i in range(len(intermediate_steps) + 1, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.show()

# Create intermediate results for visualization
X_start = X_imputed[numerical_cols].fillna(X_imputed[numerical_cols].mean())

# Apply steps individually
imputer = MissingValueImputer(strategy='mean')
X_step1 = imputer.fit_transform(X_start)

outlier_remover = OutlierRemover(method='iqr')
X_step2 = outlier_remover.fit_transform(X_step1)

scaler = ScalerTransformer(method='standard')
X_step3 = scaler.fit_transform(X_step2)

# Visualize pipeline effects
visualize_pipeline_effects(
    X_start.values,
    [X_step1, X_step2, X_step3],
    ['Imputation', 'Outlier Removal', 'Scaling']
)
```

## Advanced Preprocessing Techniques

### Handling High Cardinality Categorical Features

```python
# Simulate high cardinality categorical feature
np.random.seed(42)
high_cardinality_categories = [f'category_{i}' for i in range(100)]
high_card_feature = np.random.choice(high_cardinality_categories, size=len(df))

# Frequency encoding for high cardinality
def frequency_encode(series):
    """Encode categorical variable by frequency."""
    freq_map = series.value_counts().to_dict()
    return series.map(freq_map)

# Apply frequency encoding
freq_encoded = frequency_encode(pd.Series(high_card_feature))

print("High Cardinality Handling:")
print(f"Unique categories: {len(np.unique(high_card_feature))}")
print(f"Frequency encoding range: [{freq_encoded.min()}, {freq_encoded.max()}]")
```

### Feature Scaling with Outlier Robustness

```python
def robust_scale_with_outliers(X, outlier_method='iqr', scale_method='robust'):
    """Combine outlier removal with robust scaling."""

    # Step 1: Identify outliers
    outlier_remover = OutlierRemover(method=outlier_method)
    outlier_mask = outlier_remover.fit(X).get_outlier_mask()

    # Step 2: Fit scaler on non-outlier data
    scaler = ScalerTransformer(method=scale_method)
    X_clean = X[~outlier_mask]
    scaler.fit(X_clean)

    # Step 3: Transform all data (including outliers)
    X_scaled = scaler.transform(X)

    return X_scaled, outlier_mask

# Apply robust scaling
X_robust_scaled, outlier_mask = robust_scale_with_outliers(X_no_outliers_iqr)

print("Robust Scaling with Outlier Handling:")
print(f"Outliers identified: {outlier_mask.sum()}")
print(f"Scaling applied to all data points")
print(f"Scaled data range: [{X_robust_scaled.min():.2f}, {X_robust_scaled.max():.2f}]")
```

### Time-Based Feature Engineering

```python
# Simulate time-based features
dates = pd.date_range('2020-01-01', periods=len(df), freq='D')
df_with_time = df.copy()
df_with_time['date'] = dates

def extract_time_features(date_series):
    """Extract time-based features from datetime."""
    features = pd.DataFrame()
    features['year'] = date_series.dt.year
    features['month'] = date_series.dt.month
    features['day'] = date_series.dt.day
    features['dayofweek'] = date_series.dt.dayofweek
    features['quarter'] = date_series.dt.quarter
    features['is_weekend'] = (date_series.dt.dayofweek >= 5).astype(int)
    features['days_since_start'] = (date_series - date_series.min()).dt.days

    return features

# Extract time features
time_features = extract_time_features(df_with_time['date'])

print("Time-based Feature Engineering:")
print(f"Original date column -> {time_features.shape[1]} features")
print("Generated features:", time_features.columns.tolist())
print("\nSample time features:")
print(time_features.head())
```

## Best Practices

### 1. Data Leakage Prevention

```python
def prevent_data_leakage_example():
    """Demonstrate proper train/test splitting to prevent data leakage."""

    # WRONG: Preprocessing before splitting
    # X_processed = scaler.fit_transform(X)  # Leakage!
    # X_train, X_test = train_test_split(X_processed)

    # CORRECT: Split first, then preprocess
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X_imputed[numerical_cols], X_imputed['target'],
        test_size=0.2, random_state=42
    )

    # Fit preprocessor on training data only
    preprocessor = Pipeline([
        ('imputer', MissingValueImputer(strategy='mean')),
        ('scaler', ScalerTransformer(method='standard'))
    ])

    # Fit on training data
    X_train_processed = preprocessor.fit_transform(X_train)

    # Transform test data using fitted preprocessor
    X_test_processed = preprocessor.transform(X_test)

    print("Proper Train/Test Preprocessing:")
    print(f"Training set shape: {X_train_processed.shape}")
    print(f"Test set shape: {X_test_processed.shape}")
    print(f"Training mean: {X_train_processed.mean(axis=0)[:3]}")
    print(f"Test mean: {X_test_processed.mean(axis=0)[:3]}")

    return X_train_processed, X_test_processed, y_train, y_test

X_train_proc, X_test_proc, y_train, y_test = prevent_data_leakage_example()
```

### 2. Cross-Validation with Preprocessing

```python
def cross_validate_with_preprocessing():
    """Proper cross-validation with preprocessing inside the CV loop."""
    from sklearn.model_selection import cross_val_score
    from sklearn.ensemble import RandomForestClassifier

    # Create pipeline that includes preprocessing
    full_pipeline = Pipeline([
        ('preprocessor', Pipeline([
            ('imputer', MissingValueImputer(strategy='mean')),
            ('scaler', ScalerTransformer(method='standard')),
            ('selector', FeatureSelector(method='univariate', k=5))
        ])),
        ('classifier', RandomForestClassifier(n_estimators=50, random_state=42))
    ])

    # Cross-validation with preprocessing inside each fold
    cv_scores = cross_val_score(
        full_pipeline,
        X_imputed[numerical_cols],
        X_imputed['target'].fillna(X_imputed['target'].mode()[0]),
        cv=5,
        scoring='accuracy'
    )

    print("Cross-Validation with Preprocessing:")
    print(f"CV Scores: {cv_scores}")
    print(f"Mean CV Score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

    return cv_scores

cv_scores = cross_validate_with_preprocessing()
```

### 3. Feature Engineering Validation

```python
def validate_feature_engineering(X_original, X_engineered, feature_names):
    """Validate that feature engineering improves model performance."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score

    # Prepare target
    y = X_imputed['target'].fillna(X_imputed['target'].mode()[0])

    # Model with original features
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    scores_original = cross_val_score(model, X_original, y, cv=5)

    # Model with engineered features
    scores_engineered = cross_val_score(model, X_engineered, y, cv=5)

    print("Feature Engineering Validation:")
    print(f"Original features performance: {scores_original.mean():.3f} (+/- {scores_original.std() * 2:.3f})")
    print(f"Engineered features performance: {scores_engineered.mean():.3f} (+/- {scores_engineered.std() * 2:.3f})")

    improvement = scores_engineered.mean() - scores_original.mean()
    print(f"Performance improvement: {improvement:.3f}")

    return improvement > 0

# Validate our preprocessing pipeline
improvement = validate_feature_engineering(
    X_scaled_standard,
    X_advanced,
    numerical_cols
)
print(f"Feature engineering improved performance: {improvement}")
```

### 4. Preprocessing Documentation

```python
def document_preprocessing_pipeline(pipeline):
    """Document preprocessing pipeline for reproducibility."""

    doc = {
        'pipeline_steps': [],
        'parameters': {},
        'data_shapes': {},
        'removed_features': [],
        'created_features': []
    }

    # Extract information from pipeline
    for step_name, transformer in pipeline.steps:
        step_info = {
            'name': step_name,
            'transformer': type(transformer).__name__,
            'parameters': transformer.get_params()
        }
        doc['pipeline_steps'].append(step_info)

    # Save documentation
    import json
    with open('results/preprocessing_documentation.json', 'w') as f:
        json.dump(doc, f, indent=2, default=str)

    print("Preprocessing Pipeline Documentation:")
    print(json.dumps(doc, indent=2, default=str))

    return doc

# Document our advanced pipeline
pipeline_doc = document_preprocessing_pipeline(advanced_pipeline)
```

## Summary

This comprehensive preprocessing tutorial covered:

✅ **Data Quality Assessment** - Identifying and understanding data issues  
✅ **Missing Value Handling** - Multiple imputation strategies  
✅ **Outlier Detection** - IQR, Z-score, and Isolation Forest methods  
✅ **Feature Scaling** - Standard, MinMax, and Robust scaling  
✅ **Categorical Encoding** - One-hot, label, and target encoding  
✅ **Feature Selection** - Variance, correlation, and statistical methods  
✅ **Data Transformation** - Polynomial, interaction, and binning features  
✅ **Pipeline Construction** - Building robust preprocessing workflows  
✅ **Advanced Techniques** - Handling complex scenarios  
✅ **Best Practices** - Preventing data leakage and ensuring reproducibility

### Key Takeaways

1. **Always split before preprocessing** to prevent data leakage
2. **Use pipelines** for reproducible and maintainable preprocessing
3. **Validate feature engineering** with cross-validation
4. **Document your pipeline** for reproducibility
5. **Choose appropriate methods** based on data characteristics
6. **Monitor preprocessing effects** with visualizations

### Next Steps

- **[Feature Engineering Tutorial](feature_engineering.md)** - Advanced feature creation techniques
- **[Model Selection Tutorial](model_selection.md)** - Choosing the right algorithm
- **[Hyperparameter Tuning](hyperparameter_tuning.md)** - Optimizing model performance

The preprocessing pipeline you've built can now be used with any machine learning algorithm in the project. Remember that good preprocessing is often more important than algorithm selection for achieving high performance!
