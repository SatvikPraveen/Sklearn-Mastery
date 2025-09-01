# Dimensionality Reduction Algorithms

**File Location:** `docs/algorithm_guides/dimensionality_reduction.md`

Dimensionality reduction techniques help reduce the number of features in datasets while preserving important information. This is crucial for visualization, noise reduction, and computational efficiency.

## Overview

Dimensionality reduction methods can be broadly categorized into:

- **Linear methods**: Project data onto lower-dimensional subspaces
- **Non-linear methods**: Capture complex, non-linear relationships
- **Feature selection**: Choose subset of original features
- **Feature extraction**: Create new features from combinations of originals

## Principal Component Analysis (PCA)

### Algorithm Details

PCA finds orthogonal directions of maximum variance in the data:

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

# Example implementation
def apply_pca(X, n_components=2):
    """
    Apply PCA dimensionality reduction

    Args:
        X: Input features (n_samples, n_features)
        n_components: Number of components to keep

    Returns:
        X_transformed: Reduced features
        explained_variance_ratio: Variance explained by each component
    """
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply PCA
    pca = PCA(n_components=n_components)
    X_transformed = pca.fit_transform(X_scaled)

    return X_transformed, pca.explained_variance_ratio_

# Usage example
X_reduced, variance_explained = apply_pca(X_train, n_components=2)
print(f"Explained variance: {variance_explained}")
print(f"Total explained variance: {sum(variance_explained):.3f}")
```

### When to Use PCA

- **Best for**: Linear relationships, Gaussian-distributed data
- **Advantages**: Interpretable, computationally efficient, no hyperparameters
- **Disadvantages**: Linear assumption, sensitive to scaling
- **Use cases**: Data visualization, noise reduction, preprocessing for ML

### Hyperparameter Guidelines

```python
# Choosing number of components
pca_full = PCA()
pca_full.fit(X_scaled)

# Plot cumulative explained variance
cumsum_variance = np.cumsum(pca_full.explained_variance_ratio_)
n_components_95 = np.argmax(cumsum_variance >= 0.95) + 1
print(f"Components needed for 95% variance: {n_components_95}")
```

## t-SNE (t-Distributed Stochastic Neighbor Embedding)

### Algorithm Details

t-SNE preserves local structure by modeling pairwise similarities:

```python
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def apply_tsne(X, perplexity=30, n_components=2, random_state=42):
    """
    Apply t-SNE for non-linear dimensionality reduction

    Args:
        X: Input features
        perplexity: Balance between local and global structure
        n_components: Usually 2 or 3 for visualization

    Returns:
        X_embedded: Low-dimensional embedding
    """
    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        random_state=random_state,
        n_iter=1000
    )

    return tsne.fit_transform(X)

# Usage with different perplexities
perplexities = [5, 30, 50, 100]
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

for i, perp in enumerate(perplexities):
    X_tsne = apply_tsne(X_scaled, perplexity=perp)
    ax = axes[i//2, i%2]
    scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis')
    ax.set_title(f'Perplexity: {perp}')
    plt.colorbar(scatter, ax=ax)
```

### When to Use t-SNE

- **Best for**: Visualization, non-linear data, preserving local structure
- **Advantages**: Excellent for visualization, handles non-linear relationships
- **Disadvantages**: Stochastic, not deterministic, computationally expensive
- **Use cases**: Data exploration, cluster visualization, anomaly detection

## UMAP (Uniform Manifold Approximation and Projection)

### Algorithm Details

UMAP preserves both local and global structure:

```python
# Note: Requires umap-learn package
# pip install umap-learn

import umap

def apply_umap(X, n_neighbors=15, min_dist=0.1, n_components=2):
    """
    Apply UMAP dimensionality reduction

    Args:
        X: Input features
        n_neighbors: Local neighborhood size
        min_dist: Minimum distance in embedding
        n_components: Embedding dimensions

    Returns:
        X_embedded: UMAP embedding
    """
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        random_state=42
    )

    return reducer.fit_transform(X)

# Compare UMAP with different parameters
n_neighbors_list = [5, 15, 50, 100]
min_dist_list = [0.01, 0.1, 0.5, 0.9]

# Grid search for optimal parameters
best_score = -1
best_params = {}

for n_neigh in n_neighbors_list:
    for min_d in min_dist_list:
        X_umap = apply_umap(X_scaled, n_neighbors=n_neigh, min_dist=min_d)
        # Use silhouette score to evaluate clustering quality
        from sklearn.metrics import silhouette_score
        score = silhouette_score(X_umap, y)
        if score > best_score:
            best_score = score
            best_params = {'n_neighbors': n_neigh, 'min_dist': min_d}

print(f"Best UMAP parameters: {best_params}")
```

## Linear Discriminant Analysis (LDA)

### Algorithm Details

LDA finds linear combinations that maximize class separation:

```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def apply_lda(X, y, n_components=None):
    """
    Apply LDA for supervised dimensionality reduction

    Args:
        X: Input features
        y: Target labels
        n_components: Number of discriminant components

    Returns:
        X_transformed: LDA-transformed features
        lda_model: Fitted LDA model
    """
    if n_components is None:
        n_components = len(np.unique(y)) - 1

    lda = LinearDiscriminantAnalysis(n_components=n_components)
    X_transformed = lda.fit_transform(X, y)

    return X_transformed, lda

# Example usage
X_lda, lda_model = apply_lda(X_train, y_train)

# Visualize class separation
plt.figure(figsize=(10, 6))
for class_label in np.unique(y_train):
    mask = y_train == class_label
    plt.scatter(X_lda[mask, 0], X_lda[mask, 1],
                label=f'Class {class_label}', alpha=0.7)
plt.xlabel('LD1')
plt.ylabel('LD2')
plt.legend()
plt.title('LDA: Maximizing Class Separation')
```

## Feature Selection Methods

### Univariate Selection

```python
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif

def univariate_selection(X, y, k=10, score_func=f_classif):
    """
    Select k best features using univariate statistical tests

    Args:
        X: Input features
        y: Target variable
        k: Number of features to select
        score_func: Statistical test function

    Returns:
        X_selected: Selected features
        selector: Fitted selector object
    """
    selector = SelectKBest(score_func=score_func, k=k)
    X_selected = selector.fit_transform(X, y)

    # Get feature importance scores
    feature_scores = selector.scores_
    selected_features = selector.get_support(indices=True)

    return X_selected, selector, feature_scores, selected_features

# Compare different scoring functions
scoring_functions = {
    'f_classif': f_classif,
    'mutual_info': mutual_info_classif
}

for name, func in scoring_functions.items():
    X_sel, sel, scores, features = univariate_selection(X_train, y_train,
                                                       score_func=func)
    print(f"{name}: Selected features {features}")
    print(f"Feature scores: {scores[features]}")
```

### Recursive Feature Elimination (RFE)

```python
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

def recursive_feature_elimination(X, y, estimator=None, n_features=10):
    """
    Recursively eliminate features based on model importance

    Args:
        X: Input features
        y: Target variable
        estimator: ML model for feature ranking
        n_features: Number of features to select

    Returns:
        X_selected: Selected features
        rfe_selector: Fitted RFE selector
    """
    if estimator is None:
        estimator = RandomForestClassifier(n_estimators=100, random_state=42)

    rfe = RFE(estimator=estimator, n_features_to_select=n_features)
    X_selected = rfe.fit_transform(X, y)

    return X_selected, rfe

# Usage example
X_rfe, rfe_selector = recursive_feature_elimination(X_train, y_train, n_features=5)
selected_features = rfe_selector.get_support(indices=True)
print(f"RFE selected features: {selected_features}")
```

## Algorithm Comparison Framework

### Comprehensive Comparison

```python
import time
from sklearn.metrics import silhouette_score
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

def compare_dimensionality_reduction(X, y, methods=None):
    """
    Compare different dimensionality reduction methods

    Args:
        X: Input features
        y: Target labels
        methods: Dictionary of methods to compare

    Returns:
        results: Comparison results
    """
    if methods is None:
        methods = {
            'PCA': lambda X: PCA(n_components=2).fit_transform(X),
            'LDA': lambda X: LinearDiscriminantAnalysis(n_components=2).fit_transform(X, y),
            't-SNE': lambda X: TSNE(n_components=2, random_state=42).fit_transform(X),
        }

    results = {}

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    for name, method in methods.items():
        print(f"Evaluating {name}...")
        start_time = time.time()

        try:
            if name == 'LDA':
                X_transformed = method(X_scaled)
            else:
                X_transformed = method(X_scaled)

            # Calculate metrics
            runtime = time.time() - start_time
            silhouette = silhouette_score(X_transformed, y)

            # Evaluate downstream classification performance
            rf = RandomForestClassifier(n_estimators=50, random_state=42)
            cv_scores = cross_val_score(rf, X_transformed, y, cv=5)

            results[name] = {
                'runtime': runtime,
                'silhouette_score': silhouette,
                'classification_accuracy': cv_scores.mean(),
                'classification_std': cv_scores.std()
            }

        except Exception as e:
            print(f"Error with {name}: {e}")
            results[name] = {'error': str(e)}

    return results

# Run comparison
comparison_results = compare_dimensionality_reduction(X_train, y_train)

# Display results
print("\nDimensionality Reduction Comparison:")
print("-" * 60)
for method, metrics in comparison_results.items():
    if 'error' not in metrics:
        print(f"{method:10s} | Runtime: {metrics['runtime']:.3f}s | "
              f"Silhouette: {metrics['silhouette_score']:.3f} | "
              f"Accuracy: {metrics['classification_accuracy']:.3f}±{metrics['classification_std']:.3f}")
```

## Best Practices and Guidelines

### Choosing the Right Method

1. **For Linear Data**:

   - Use PCA for noise reduction and computational efficiency
   - Use LDA when you have labeled data and want to maximize class separation

2. **For Non-Linear Data**:

   - Use t-SNE for visualization (2D/3D only)
   - Use UMAP for both visualization and further processing
   - Consider autoencoders for very high-dimensional data

3. **For Feature Selection**:
   - Use univariate selection for quick filtering
   - Use RFE when computational resources allow
   - Use L1 regularization (Lasso) for sparse feature selection

### Preprocessing Considerations

```python
# Always standardize features for PCA, t-SNE, UMAP
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# For PCA and LDA
scaler = StandardScaler()  # Zero mean, unit variance

# For t-SNE and UMAP (can also use MinMaxScaler)
scaler = MinMaxScaler()  # Scale to [0,1] range

X_scaled = scaler.fit_transform(X)
```

### Evaluation Metrics

```python
def evaluate_dimensionality_reduction(X_original, X_reduced, y):
    """
    Comprehensive evaluation of dimensionality reduction

    Args:
        X_original: Original high-dimensional data
        X_reduced: Reduced-dimensional data
        y: Target labels

    Returns:
        metrics: Dictionary of evaluation metrics
    """
    from sklearn.metrics import silhouette_score
    from sklearn.neighbors import NearestNeighbors
    import numpy as np

    metrics = {}

    # Clustering quality
    if len(np.unique(y)) > 1:
        metrics['silhouette_score'] = silhouette_score(X_reduced, y)

    # Neighborhood preservation (for non-linear methods)
    if X_original.shape[1] > X_reduced.shape[1]:
        k = min(10, X_original.shape[0] - 1)

        # Original space neighbors
        nbrs_orig = NearestNeighbors(n_neighbors=k).fit(X_original)
        _, indices_orig = nbrs_orig.kneighbors(X_original)

        # Reduced space neighbors
        nbrs_red = NearestNeighbors(n_neighbors=k).fit(X_reduced)
        _, indices_red = nbrs_red.kneighbors(X_reduced)

        # Calculate neighborhood preservation
        preservation = []
        for i in range(len(indices_orig)):
            intersection = len(set(indices_orig[i]) & set(indices_red[i]))
            preservation.append(intersection / k)

        metrics['neighborhood_preservation'] = np.mean(preservation)

    # Explained variance (for linear methods like PCA)
    metrics['variance_retained'] = 'N/A'  # Method-specific

    return metrics
```

This comprehensive guide covers the major dimensionality reduction techniques, their implementations, when to use each method, and how to evaluate their performance. Choose the method that best fits your data characteristics and downstream tasks.
