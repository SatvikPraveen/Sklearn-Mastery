# File: docs/algorithm_guides/clustering.md

# Location: docs/algorithm_guides/clustering.md

# Clustering Algorithms

Comprehensive guide to unsupervised clustering algorithms in the ML Pipeline Framework.

## Overview

Clustering algorithms group similar data points without labeled targets. Our framework provides optimized implementations with automatic parameter selection and cluster validation.

## Available Algorithms

### K-Means Clustering

**Best for**: Spherical clusters, known number of clusters, large datasets

```python
from src.models.unsupervised.clustering import ClusteringModels

models = ClusteringModels()
kmeans = models.get_kmeans()

# Optimized configuration
kmeans = models.get_kmeans(
    n_clusters=5,
    init='k-means++',
    n_init=10,
    max_iter=300,
    random_state=42
)
```

**When to use**:

- Know approximate number of clusters
- Clusters are roughly spherical
- Similar cluster sizes expected
- Need fast, scalable algorithm

**Key parameters**:

- `n_clusters`: Number of clusters (required)
- `init`: Initialization method ('k-means++' recommended)
- `n_init`: Number of random initializations (10 default)
- `max_iter`: Maximum iterations (300 default)

### DBSCAN

**Best for**: Arbitrary shaped clusters, outlier detection, unknown cluster count

```python
dbscan = models.get_dbscan()

# Fine-tuned for density
dbscan = models.get_dbscan(
    eps=0.5,
    min_samples=5,
    metric='euclidean',
    algorithm='auto'
)
```

**When to use**:

- Don't know number of clusters
- Expect outliers in data
- Clusters have varying densities
- Non-spherical cluster shapes

**Parameter tuning**:

- `eps`: Maximum distance between points (critical parameter)
- `min_samples`: Minimum points in neighborhood (5-10 typical)
- Use k-distance graph to find optimal `eps`

### Hierarchical Clustering

**Best for**: Cluster hierarchies, small to medium datasets, dendrograms

```python
hierarchical = models.get_hierarchical()

# Complete linkage for compact clusters
hierarchical = models.get_hierarchical(
    n_clusters=4,
    linkage='complete',
    distance_threshold=None,
    affinity='euclidean'
)
```

**When to use**:

- Need cluster hierarchy/dendrogram
- Small to medium datasets (< 10K points)
- Want to explore different cluster counts
- Interpretable cluster relationships

**Linkage methods**:

- `complete`: Maximum distance (compact clusters)
- `average`: Average distance (balanced)
- `single`: Minimum distance (can chain)
- `ward`: Minimize variance (spherical clusters)

### Gaussian Mixture Models (GMM)

**Best for**: Overlapping clusters, probabilistic assignments, soft clustering

```python
gmm = models.get_gaussian_mixture()

# Full covariance for flexible shapes
gmm = models.get_gaussian_mixture(
    n_components=3,
    covariance_type='full',
    init_params='kmeans',
    max_iter=200
)
```

**When to use**:

- Overlapping clusters expected
- Need probabilistic cluster membership
- Clusters have different shapes/orientations
- Want soft cluster assignments

**Covariance types**:

- `full`: General elliptical clusters (flexible)
- `tied`: Same shape for all clusters
- `diag`: Axis-aligned ellipses
- `spherical`: Circular clusters (fastest)

### Spectral Clustering

**Best for**: Non-convex clusters, manifold data, graph-based clustering

```python
spectral = models.get_spectral_clustering()

# RBF kernel for complex shapes
spectral = models.get_spectral_clustering(
    n_clusters=3,
    affinity='rbf',
    gamma=1.0,
    n_neighbors=10
)
```

**When to use**:

- Complex, non-convex cluster shapes
- Data lies on manifolds
- Graph-based relationships
- Need to find connected components

**Affinity options**:

- `rbf`: RBF kernel (general purpose)
- `nearest_neighbors`: k-NN graph
- `precomputed`: Custom similarity matrix

### Mean Shift

**Best for**: Variable cluster sizes, peak detection, bandwidth estimation

```python
meanshift = models.get_mean_shift()

# Auto bandwidth estimation
meanshift = models.get_mean_shift(
    bandwidth=None,  # Auto-estimate
    bin_seeding=True,
    cluster_all=True
)
```

**When to use**:

- Don't know number of clusters
- Clusters have varying sizes
- Want mode-seeking behavior
- Peak detection in density

## Performance Comparison

| Algorithm    | Speed      | Scalability | Cluster Shapes | Outlier Handling |
| ------------ | ---------- | ----------- | -------------- | ---------------- |
| K-Means      | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐  | Spherical      | ❌               |
| DBSCAN       | ⭐⭐⭐     | ⭐⭐⭐⭐    | Arbitrary      | ⭐⭐⭐⭐⭐       |
| Hierarchical | ⭐⭐       | ⭐⭐        | Arbitrary      | ⭐⭐             |
| GMM          | ⭐⭐⭐     | ⭐⭐⭐      | Elliptical     | ⭐⭐⭐           |
| Spectral     | ⭐⭐       | ⭐⭐        | Complex        | ⭐⭐             |
| Mean Shift   | ⭐         | ⭐⭐        | Arbitrary      | ⭐⭐⭐           |

## Usage Examples

### Basic Clustering Pipeline

```python
from src.data.generators import DataGenerator
from src.models.unsupervised.clustering import ClusteringModels
from src.evaluation.metrics import ClusteringEvaluator

# Generate sample data
generator = DataGenerator()
X, true_labels = generator.generate_clustering_data(
    n_samples=1000,
    n_features=2,
    n_clusters=4,
    cluster_std=1.0,
    random_state=42
)

# Test multiple algorithms
models = ClusteringModels()
evaluator = ClusteringEvaluator()

algorithms = [
    ('K-Means', models.get_kmeans(n_clusters=4)),
    ('DBSCAN', models.get_dbscan(eps=0.8, min_samples=5)),
    ('Hierarchical', models.get_hierarchical(n_clusters=4)),
    ('GMM', models.get_gaussian_mixture(n_components=4))
]

results = {}
for name, model in algorithms:
    # Fit and predict
    cluster_labels = model.fit_predict(X)

    # Evaluate clustering quality
    metrics = evaluator.evaluate_clustering(X, cluster_labels, true_labels)
    results[name] = metrics

# Display results
for name, metrics in results.items():
    print(f"\n{name} Results:")
    print(f"  Silhouette Score: {metrics['silhouette_score']:.3f}")
    print(f"  Adjusted Rand Index: {metrics['adjusted_rand_score']:.3f}")
    print(f"  Homogeneity: {metrics['homogeneity_score']:.3f}")
```

### Optimal Cluster Number Detection

```python
from src.evaluation.utils import OptimalClusters

# Elbow method for K-Means
optimal_k = OptimalClusters()

# Test range of cluster numbers
k_range = range(2, 11)
inertias = optimal_k.elbow_method(X, k_range)
optimal_clusters = optimal_k.find_elbow_point(k_range, inertias)

print(f"Optimal clusters (Elbow): {optimal_clusters}")

# Silhouette analysis
silhouette_scores = optimal_k.silhouette_analysis(X, k_range)
best_k_silhouette = k_range[np.argmax(silhouette_scores)]

print(f"Optimal clusters (Silhouette): {best_k_silhouette}")

# Gap statistic
gap_stats, std_gaps = optimal_k.gap_statistic(X, k_range, n_refs=10)
optimal_k_gap = optimal_k.find_gap_optimal(gap_stats, std_gaps)

print(f"Optimal clusters (Gap): {optimal_k_gap}")
```

### DBSCAN Parameter Tuning

```python
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

# Find optimal eps using k-distance graph
def find_optimal_eps(X, k=5):
    neighbors = NearestNeighbors(n_neighbors=k)
    neighbors.fit(X)
    distances, indices = neighbors.kneighbors(X)

    # Sort distances to k-th nearest neighbor
    k_distances = np.sort(distances[:, k-1])

    # Plot k-distance graph
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(k_distances)), k_distances)
    plt.xlabel('Data Points')
    plt.ylabel(f'{k}-Distance')
    plt.title('K-Distance Graph for DBSCAN eps Selection')
    plt.grid(True)

    # Find knee/elbow in the curve
    # Simple approach: look for maximum curvature
    diffs = np.diff(k_distances)
    knee_idx = np.argmax(diffs) + 1
    optimal_eps = k_distances[knee_idx]

    plt.axhline(y=optimal_eps, color='r', linestyle='--',
                label=f'Suggested eps: {optimal_eps:.3f}')
    plt.legend()
    plt.show()

    return optimal_eps

# Find optimal parameters
optimal_eps = find_optimal_eps(X, k=5)
dbscan_tuned = models.get_dbscan(eps=optimal_eps, min_samples=5)
labels = dbscan_tuned.fit_predict(X)

print(f"Clusters found: {len(set(labels)) - (1 if -1 in labels else 0)}")
print(f"Outliers: {list(labels).count(-1)}")
```

## Advanced Techniques

### Ensemble Clustering

```python
from src.models.ensemble.clustering import EnsembleClustering

# Combine multiple clustering algorithms
ensemble = EnsembleClustering()

base_clusterers = [
    models.get_kmeans(n_clusters=4),
    models.get_hierarchical(n_clusters=4),
    models.get_gaussian_mixture(n_components=4)
]

# Consensus clustering
final_labels = ensemble.consensus_clustering(
    X, base_clusterers, method='majority_vote'
)

# Evidence accumulation
evidence_labels = ensemble.evidence_accumulation(
    X, base_clusterers, n_consensus_clusters=4
)
```

### Dimensionality Reduction + Clustering

```python
from src.models.unsupervised.dimensionality_reduction import DimensionalityReduction

# Reduce dimensions before clustering
dim_reducer = DimensionalityReduction()
clusterer = models.get_kmeans(n_clusters=4)

# PCA + K-Means
pca = dim_reducer.get_pca(n_components=10)
X_reduced = pca.fit_transform(X)
labels = clusterer.fit_predict(X_reduced)

# t-SNE + DBSCAN (good for visualization)
tsne = dim_reducer.get_tsne(n_components=2)
X_tsne = tsne.fit_transform(X)
dbscan = models.get_dbscan(eps=0.5)
labels_tsne = dbscan.fit_predict(X_tsne)
```

### Custom Distance Metrics

```python
# Custom distance function
def custom_distance(x1, x2):
    # Example: weighted Euclidean distance
    weights = np.array([2.0, 1.0, 0.5])  # Feature weights
    return np.sqrt(np.sum(weights * (x1 - x2) ** 2))

# Use with hierarchical clustering
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, fcluster

# Compute distance matrix
distances = pdist(X, metric=custom_distance)
linkage_matrix = linkage(distances, method='complete')
labels = fcluster(linkage_matrix, t=4, criterion='maxclust')
```

## Evaluation Metrics

### Internal Metrics (No Ground Truth)

```python
from src.evaluation.metrics import ClusteringMetrics

metrics = ClusteringMetrics()

# Silhouette Score (-1 to 1, higher is better)
silhouette = metrics.silhouette_score(X, labels)

# Calinski-Harabasz Index (higher is better)
ch_score = metrics.calinski_harabasz_score(X, labels)

# Davies-Bouldin Index (lower is better)
db_score = metrics.davies_bouldin_score(X, labels)

print(f"Silhouette Score: {silhouette:.3f}")
print(f"Calinski-Harabasz: {ch_score:.3f}")
print(f"Davies-Bouldin: {db_score:.3f}")
```

### External Metrics (With Ground Truth)

```python
# Adjusted Rand Index (0 to 1, higher is better)
ari = metrics.adjusted_rand_score(true_labels, predicted_labels)

# Normalized Mutual Information (0 to 1, higher is better)
nmi = metrics.normalized_mutual_info_score(true_labels, predicted_labels)

# Homogeneity and Completeness
homogeneity = metrics.homogeneity_score(true_labels, predicted_labels)
completeness = metrics.completeness_score(true_labels, predicted_labels)
v_measure = metrics.v_measure_score(true_labels, predicted_labels)

print(f"Adjusted Rand Index: {ari:.3f}")
print(f"Normalized MI: {nmi:.3f}")
print(f"Homogeneity: {homogeneity:.3f}")
print(f"Completeness: {completeness:.3f}")
print(f"V-Measure: {v_measure:.3f}")
```

## Visualization

### Cluster Visualization

```python
from src.evaluation.visualization import ClusteringVisualizer

visualizer = ClusteringVisualizer()

# 2D scatter plot
visualizer.plot_clusters_2d(X, labels, title="Clustering Results")

# 3D visualization for 3D data
if X.shape[1] >= 3:
    visualizer.plot_clusters_3d(X[:, :3], labels, title="3D Clustering")

# Cluster centers (for centroid-based methods)
if hasattr(model, 'cluster_centers_'):
    visualizer.plot_cluster_centers(X, labels, model.cluster_centers_)

# Dendrogram for hierarchical clustering
if isinstance(model, AgglomerativeClustering):
    visualizer.plot_dendrogram(X, method='complete')
```

### Performance Comparison Plots

```python
# Compare multiple algorithms
comparison_results = {}
algorithms = ['K-Means', 'DBSCAN', 'Hierarchical', 'GMM']

for algo_name in algorithms:
    # ... run clustering and evaluation ...
    comparison_results[algo_name] = {
        'silhouette': silhouette_score,
        'ari': ari_score,
        'time': execution_time
    }

# Plot comparison
visualizer.plot_algorithm_comparison(
    comparison_results,
    metrics=['silhouette', 'ari'],
    title="Clustering Algorithm Comparison"
)
```

## Best Practices

### Algorithm Selection Guide

**Data characteristics**:

- **Known clusters**: K-Means, GMM, Hierarchical
- **Unknown clusters**: DBSCAN, Mean Shift
- **Overlapping data**: GMM, Spectral
- **Outliers present**: DBSCAN, Mean Shift
- **Large datasets**: K-Means, Mini-Batch K-Means
- **Complex shapes**: DBSCAN, Spectral, Hierarchical

**Performance tips**:

1. **Preprocessing**: Scale features, handle outliers
2. **Validation**: Use multiple internal metrics
3. **Stability**: Run multiple times with different seeds
4. **Domain knowledge**: Incorporate prior knowledge where possible

### Common Pitfalls

**K-Means limitations**:

- Assumes spherical clusters
- Sensitive to initialization and outliers
- Requires pre-specified k

**DBSCAN challenges**:

- Parameter selection (eps, min_samples)
- Struggles with varying densities
- Memory intensive for large datasets

**Solutions**:

- Use ensemble methods
- Validate with multiple metrics
- Consider data preprocessing
- Try parameter sensitivity analysis

## See Also

- [Dimensionality Reduction](dimensionality_reduction.md)
- [Data Preprocessing](../tutorials/data_preprocessing.md)
- [API Reference: Clustering](../api_reference/models.md#clustering)
- [Evaluation Metrics](../api_reference/evaluation.md)
