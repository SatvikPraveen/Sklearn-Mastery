"""
Unit tests for clustering models.

Tests for all clustering model wrappers and utilities.
"""

import pytest
import numpy as np
from sklearn.datasets import make_blobs, make_circles
from sklearn.metrics import adjusted_rand_score, silhouette_score
import sys
import os
import tempfile

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from models.unsupervised.clustering import (
    ClusteringModel,
    KMeansModel,
    HierarchicalClusteringModel,
    DBSCANModel,
    GaussianMixtureModel,
    SpectralClusteringModel,
    AffinityPropagationModel,
    MeanShiftModel,
    BirchModel,
    MiniBatchKMeansModel,
    OPTICSModel
)


class TestClusteringModel:
    """Test base ClusteringModel class."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample clustering data."""
        X, y = make_blobs(
            n_samples=200,
            centers=3,
            n_features=5,
            random_state=42
        )
        return X, y
    
    def test_abstract_methods(self):
        """Test that abstract methods raise NotImplementedError."""
        model = ClusteringModel()
        
        with pytest.raises(NotImplementedError):
            model.fit(None)
        
        with pytest.raises(NotImplementedError):
            model.predict(None)
    
    def test_evaluate_method(self, sample_data):
        """Test evaluate method with a concrete implementation."""
        X, y_true = sample_data
        
        # Use a concrete implementation
        model = KMeansModel(n_clusters=3, random_state=42)
        model.fit(X)
        y_pred = model.predict(X)
        
        metrics = model.evaluate(X, y_true, y_pred)
        
        assert 'silhouette_score' in metrics
        assert 'adjusted_rand_score' in metrics
        assert 'calinski_harabasz_score' in metrics
        assert 'davies_bouldin_score' in metrics
        
        # Check metric ranges
        assert -1 <= metrics['silhouette_score'] <= 1
        assert -1 <= metrics['adjusted_rand_score'] <= 1
        assert metrics['calinski_harabasz_score'] >= 0
        assert metrics['davies_bouldin_score'] >= 0


class TestKMeansModel:
    """Test KMeansModel class."""
    
    @pytest.fixture
    def model(self):
        """Create model instance."""
        return KMeansModel(n_clusters=3, random_state=42)
    
    @pytest.fixture
    def data(self):
        """Generate test data."""
        X, y = make_blobs(
            n_samples=100,
            centers=3,
            n_features=5,
            random_state=42
        )
        return X, y
    
    def test_initialization(self, model):
        """Test model initialization."""
        assert model.n_clusters == 3
        assert model.random_state == 42
        assert hasattr(model, 'model')
    
    def test_fit_predict(self, model, data):
        """Test fitting and prediction."""
        X, y = data
        
        model.fit(X)
        predictions = model.predict(X)
        
        assert len(predictions) == len(y)
        # Number of clusters is determined automatically
        assert len(set(predictions)) >= 1
    
    def test_cluster_centers(self, model, data):
        """Test cluster centers extraction."""
        X, y = data
        model.fit(X)
        
        centers = model.get_cluster_centers()
        
        assert centers.shape[1] == X.shape[1]
        assert centers.shape[0] >= 1


class TestBirchModel:
    """Test BirchModel class."""
    
    @pytest.fixture
    def model(self):
        """Create model instance."""
        return BirchModel(n_clusters=3)
    
    @pytest.fixture
    def data(self):
        """Generate test data."""
        X, y = make_blobs(
            n_samples=100,
            centers=3,
            n_features=5,
            random_state=42
        )
        return X, y
    
    def test_initialization(self, model):
        """Test model initialization."""
        assert model.n_clusters == 3
        assert hasattr(model, 'model')
    
    def test_fit_predict(self, model, data):
        """Test fitting and prediction."""
        X, y = data
        
        model.fit(X)
        predictions = model.predict(X)
        
        assert len(predictions) == len(y)
        assert len(set(predictions)) <= model.n_clusters
    
    def test_transform(self, model, data):
        """Test feature transformation."""
        X, y = data
        model.fit(X)
        
        X_transformed = model.transform(X)
        
        assert X_transformed.shape[0] == X.shape[0]
        # Transformed features should be <= original features
        assert X_transformed.shape[1] <= X.shape[1]


class TestMiniBatchKMeansModel:
    """Test MiniBatchKMeansModel class."""
    
    @pytest.fixture
    def model(self):
        """Create model instance."""
        return MiniBatchKMeansModel(n_clusters=3, random_state=42, batch_size=20)
    
    @pytest.fixture
    def data(self):
        """Generate test data."""
        X, y = make_blobs(
            n_samples=200,
            centers=3,
            n_features=5,
            random_state=42
        )
        return X, y
    
    def test_initialization(self, model):
        """Test model initialization."""
        assert model.n_clusters == 3
        assert model.random_state == 42
        assert model.batch_size == 20
        assert hasattr(model, 'model')
    
    def test_fit_predict(self, model, data):
        """Test fitting and prediction."""
        X, y = data
        
        model.fit(X)
        predictions = model.predict(X)
        
        assert len(predictions) == len(y)
        assert len(set(predictions)) <= model.n_clusters
    
    def test_partial_fit(self, model, data):
        """Test partial fitting capability."""
        X, y = data
        
        # Split data into batches
        batch_size = 50
        for i in range(0, len(X), batch_size):
            X_batch = X[i:i+batch_size]
            model.partial_fit(X_batch)
        
        predictions = model.predict(X)
        
        assert len(predictions) == len(y)
        assert len(set(predictions)) <= model.n_clusters


class TestOPTICSModel:
    """Test OPTICSModel class."""
    
    @pytest.fixture
    def model(self):
        """Create model instance."""
        return OPTICSModel(min_samples=5)
    
    @pytest.fixture
    def data(self):
        """Generate test data."""
        X, y = make_blobs(
            n_samples=100,
            centers=3,
            n_features=5,
            cluster_std=0.8,
            random_state=42
        )
        return X, y
    
    def test_initialization(self, model):
        """Test model initialization."""
        assert model.min_samples == 5
        assert hasattr(model, 'model')
    
    def test_fit_predict(self, model, data):
        """Test fitting and prediction."""
        X, y = data
        
        model.fit(X)
        predictions = model.predict(X)
        
        assert len(predictions) == len(y)
        # OPTICS can predict -1 for noise points
        assert all(pred >= -1 for pred in predictions)
    
    def test_reachability(self, model, data):
        """Test reachability distances."""
        X, y = data
        model.fit(X)
        
        reachability = model.get_reachability()
        
        assert len(reachability) == len(X)
        assert np.all(reachability >= 0)
    
    def test_ordering(self, model, data):
        """Test cluster ordering."""
        X, y = data
        model.fit(X)
        
        ordering = model.get_ordering()
        
        assert len(ordering) == len(X)
        assert set(ordering) == set(range(len(X)))


class TestClusteringModelIntegration:
    """Integration tests for clustering models."""
    
    @pytest.fixture
    def models(self):
        """Create all model instances."""
        return {
            'kmeans': KMeansModel(n_clusters=3, random_state=42),
            'hierarchical': HierarchicalClusteringModel(n_clusters=3),
            'dbscan': DBSCANModel(eps=1.0, min_samples=5),
            'gmm': GaussianMixtureModel(n_components=3, random_state=42),
            'spectral': SpectralClusteringModel(n_clusters=3, random_state=42),
            'birch': BirchModel(n_clusters=3),
            'mini_kmeans': MiniBatchKMeansModel(n_clusters=3, random_state=42)
        }
    
    @pytest.fixture
    def data(self):
        """Generate test data."""
        X, y = make_blobs(
            n_samples=150,
            centers=3,
            n_features=5,
            cluster_std=1.0,
            random_state=42
        )
        return X, y
    
    def test_all_models_fit_predict(self, models, data):
        """Test that all models can fit and predict."""
        X, y = data
        
        for name, model in models.items():
            model.fit(X)
            predictions = model.predict(X)
            
            assert len(predictions) == len(y), f"{name} failed prediction length test"
            assert len(set(predictions)) >= 1, f"{name} should find at least one cluster"
    
    def test_clustering_quality(self, models, data):
        """Test clustering quality with silhouette score."""
        X, y = data
        
        for name, model in models.items():
            model.fit(X)
            predictions = model.predict(X)
            
            # Skip if only one cluster or noise points only
            unique_labels = set(predictions)
            if len(unique_labels) <= 1 or (len(unique_labels) == 2 and -1 in unique_labels):
                continue
            
            # Calculate silhouette score
            score = silhouette_score(X, predictions)
            
            # Should be reasonable (> -0.5 for most cases)
            assert score > -0.5, f"{name} silhouette score too low: {score}"
    
    def test_different_data_shapes(self, models):
        """Test models with different data shapes."""
        # Small dataset
        X_small, _ = make_blobs(n_samples=30, centers=2, n_features=3, random_state=42)
        
        # High-dimensional dataset
        X_high_dim, _ = make_blobs(n_samples=100, centers=3, n_features=20, random_state=42)
        
        test_datasets = [
            ("small", X_small),
            ("high_dim", X_high_dim)
        ]
        
        for data_name, X in test_datasets:
            for model_name, model in models.items():
                # Skip computationally expensive combinations
                if data_name == "high_dim" and model_name in ["hierarchical", "spectral"]:
                    continue
                
                try:
                    model.fit(X)
                    predictions = model.predict(X)
                    assert len(predictions) == len(X), f"{model_name} failed on {data_name} data"
                except Exception as e:
                    pytest.fail(f"{model_name} failed on {data_name} data: {str(e)}")
    
    def test_reproducibility(self, data):
        """Test that models with random_state are reproducible."""
        X, y = data
        
        reproducible_models = [
            ('kmeans1', KMeansModel(n_clusters=3, random_state=42)),
            ('kmeans2', KMeansModel(n_clusters=3, random_state=42)),
            ('gmm1', GaussianMixtureModel(n_components=3, random_state=42)),
            ('gmm2', GaussianMixtureModel(n_components=3, random_state=42))
        ]
        
        # Test pairs of identical models
        for i in range(0, len(reproducible_models), 2):
            name1, model1 = reproducible_models[i]
            name2, model2 = reproducible_models[i + 1]
            
            model1.fit(X)
            model2.fit(X)
            
            pred1 = model1.predict(X)
            pred2 = model2.predict(X)
            
            # Results should be identical for same random state
            np.testing.assert_array_equal(
                pred1, pred2,
                f"Models {name1} and {name2} should produce identical results"
            )
    
    def test_parameter_sensitivity(self, data):
        """Test sensitivity to different parameters."""
        X, y = data
        
        # Test K-means with different cluster numbers
        k_values = [2, 3, 4, 5]
        inertias = []
        
        for k in k_values:
            model = KMeansModel(n_clusters=k, random_state=42)
            model.fit(X)
            inertias.append(model.get_inertia())
        
        # Inertia should generally decrease with more clusters
        assert inertias[0] >= inertias[-1], "Inertia should decrease with more clusters"
        
        # Test DBSCAN with different eps values
        eps_values = [0.5, 1.0, 2.0]
        n_clusters = []
        
        for eps in eps_values:
            model = DBSCANModel(eps=eps, min_samples=5)
            model.fit(X)
            predictions = model.predict(X)
            unique_clusters = len(set(predictions) - {-1})  # Exclude noise label
            n_clusters.append(unique_clusters)
        
        # Should find some clusters
        assert max(n_clusters) > 0, "DBSCAN should find at least some clusters" == len(y)
        assert len(set(predictions)) <= model.n_clusters
        assert all(0 <= pred < model.n_clusters for pred in predictions)
    
    def test_cluster_centers(self, model, data):
        """Test cluster centers extraction."""
        X, y = data
        model.fit(X)
        
        centers = model.get_cluster_centers()
        
        assert centers.shape == (model.n_clusters, X.shape[1])
        assert centers.dtype == np.float64
    
    def test_inertia(self, model, data):
        """Test inertia calculation."""
        X, y = data
        model.fit(X)
        
        inertia = model.get_inertia()
        
        assert isinstance(inertia, (float, np.float64))
        assert inertia >= 0
    
    def test_transform(self, model, data):
        """Test distance transformation."""
        X, y = data
        model.fit(X)
        
        distances = model.transform(X)
        
        assert distances.shape == (len(X), model.n_clusters)
        assert np.all(distances >= 0)
    
    def test_save_load(self, model, data):
        """Test model saving and loading."""
        X, y = data
        model.fit(X)
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = os.path.join(tmp_dir, 'test_model.pkl')
            model.save_model(model_path)
            
            # Load model and test
            loaded_model = KMeansModel(n_clusters=3)
            loaded_model.load_model(model_path)
            
            # Test predictions are the same
            original_pred = model.predict(X)
            loaded_pred = loaded_model.predict(X)
            
            np.testing.assert_array_equal(original_pred, loaded_pred)


class TestHierarchicalClusteringModel:
    """Test HierarchicalClusteringModel class."""
    
    @pytest.fixture
    def model(self):
        """Create model instance."""
        return HierarchicalClusteringModel(n_clusters=3)
    
    @pytest.fixture
    def data(self):
        """Generate test data."""
        X, y = make_blobs(
            n_samples=50,  # Smaller dataset for hierarchical clustering
            centers=3,
            n_features=5,
            random_state=42
        )
        return X, y
    
    def test_initialization(self, model):
        """Test model initialization."""
        assert model.n_clusters == 3
        assert hasattr(model, 'model')
    
    def test_fit_predict(self, model, data):
        """Test fitting and prediction."""
        X, y = data
        
        model.fit(X)
        predictions = model.predict(X)
        
        assert len(predictions) == len(y)
        assert len(set(predictions)) <= model.n_clusters
    
    def test_different_linkages(self, data):
        """Test different linkage methods."""
        X, y = data
        
        linkages = ['ward', 'complete', 'average', 'single']
        
        for linkage in linkages:
            model = HierarchicalClusteringModel(n_clusters=3, linkage=linkage)
            model.fit(X)
            predictions = model.predict(X)
            
            assert len(predictions) == len(y)
            assert len(set(predictions)) <= 3


class TestDBSCANModel:
    """Test DBSCANModel class."""
    
    @pytest.fixture
    def model(self):
        """Create model instance."""
        return DBSCANModel(eps=0.5, min_samples=5)
    
    @pytest.fixture
    def data(self):
        """Generate test data."""
        X, y = make_blobs(
            n_samples=100,
            centers=3,
            n_features=5,
            cluster_std=0.8,
            random_state=42
        )
        return X, y
    
    def test_initialization(self, model):
        """Test model initialization."""
        assert model.eps == 0.5
        assert model.min_samples == 5
        assert hasattr(model, 'model')
    
    def test_fit_predict(self, model, data):
        """Test fitting and prediction."""
        X, y = data
        
        model.fit(X)
        predictions = model.predict(X)
        
        assert len(predictions) == len(y)
        # DBSCAN can predict -1 for noise points
        assert all(pred >= -1 for pred in predictions)
    
    def test_core_samples(self, model, data):
        """Test core samples identification."""
        X, y = data
        model.fit(X)
        
        core_samples = model.get_core_samples()
        
        assert isinstance(core_samples, np.ndarray)
        assert len(core_samples) <= len(X)
        assert all(0 <= idx < len(X) for idx in core_samples)
    
    def test_noise_detection(self, data):
        """Test noise point detection."""
        X, y = data
        
        # Add some obvious outliers
        outliers = np.random.randn(5, X.shape[1]) * 10
        X_with_outliers = np.vstack([X, outliers])
        
        model = DBSCANModel(eps=1.0, min_samples=5)
        model.fit(X_with_outliers)
        predictions = model.predict(X_with_outliers)
        
        # Should detect some noise points
        assert -1 in predictions


class TestGaussianMixtureModel:
    """Test GaussianMixtureModel class."""
    
    @pytest.fixture
    def model(self):
        """Create model instance."""
        return GaussianMixtureModel(n_components=3, random_state=42)
    
    @pytest.fixture
    def data(self):
        """Generate test data."""
        X, y = make_blobs(
            n_samples=100,
            centers=3,
            n_features=5,
            random_state=42
        )
        return X, y
    
    def test_initialization(self, model):
        """Test model initialization."""
        assert model.n_components == 3
        assert model.random_state == 42
        assert hasattr(model, 'model')
    
    def test_fit_predict(self, model, data):
        """Test fitting and prediction."""
        X, y = data
        
        model.fit(X)
        predictions = model.predict(X)
        
        assert len(predictions) == len(y)
        assert len(set(predictions)) <= model.n_components
    
    def test_predict_proba(self, model, data):
        """Test probability prediction."""
        X, y = data
        model.fit(X)
        
        probas = model.predict_proba(X)
        
        assert probas.shape == (len(X), model.n_components)
        assert np.allclose(probas.sum(axis=1), 1.0)
        assert np.all((probas >= 0) & (probas <= 1))
    
    def test_log_likelihood(self, model, data):
        """Test log-likelihood calculation."""
        X, y = data
        model.fit(X)
        
        log_likelihood = model.score(X)
        
        assert isinstance(log_likelihood, (float, np.float64))
    
    def test_bic_aic(self, model, data):
        """Test BIC and AIC calculations."""
        X, y = data
        model.fit(X)
        
        bic = model.bic(X)
        aic = model.aic(X)
        
        assert isinstance(bic, (float, np.float64))
        assert isinstance(aic, (float, np.float64))


class TestSpectralClusteringModel:
    """Test SpectralClusteringModel class."""
    
    @pytest.fixture
    def model(self):
        """Create model instance."""
        return SpectralClusteringModel(n_clusters=2, random_state=42)
    
    @pytest.fixture
    def data(self):
        """Generate test data with non-convex clusters."""
        X, y = make_circles(
            n_samples=100,
            factor=0.3,
            noise=0.1,
            random_state=42
        )
        return X, y
    
    def test_initialization(self, model):
        """Test model initialization."""
        assert model.n_clusters == 2
        assert model.random_state == 42
        assert hasattr(model, 'model')
    
    def test_fit_predict(self, model, data):
        """Test fitting and prediction."""
        X, y = data
        
        model.fit(X)
        predictions = model.predict(X)
        
        assert len(predictions) == len(y)
        assert len(set(predictions)) <= model.n_clusters
    
    def test_affinity_matrix(self, model, data):
        """Test different affinity matrices."""
        X, y = data
        
        affinities = ['rbf', 'nearest_neighbors']
        
        for affinity in affinities:
            model = SpectralClusteringModel(n_clusters=2, affinity=affinity, random_state=42)
            model.fit(X)
            predictions = model.predict(X)
            
            assert len(predictions) == len(y)


class TestAffinityPropagationModel:
    """Test AffinityPropagationModel class."""
    
    @pytest.fixture
    def model(self):
        """Create model instance."""
        return AffinityPropagationModel(random_state=42)
    
    @pytest.fixture
    def data(self):
        """Generate test data."""
        X, y = make_blobs(
            n_samples=50,  # Smaller dataset for affinity propagation
            centers=3,
            n_features=5,
            random_state=42
        )
        return X, y
    
    def test_initialization(self, model):
        """Test model initialization."""
        assert model.random_state == 42
        assert hasattr(model, 'model')
    
    def test_fit_predict(self, model, data):
        """Test fitting and prediction."""
        X, y = data
        
        model.fit(X)
        predictions = model.predict(X)
        
        assert len(predictions) == len(y)
        # Number of clusters is determined automatically
        assert len(set(predictions)) >= 1
    
    def test_cluster_centers(self, model, data):
        """Test cluster centers (exemplars) extraction."""
        X, y = data
        model.fit(X)
        
        centers_indices = model.get_cluster_centers_indices()
        
        assert isinstance(centers_indices, np.ndarray)
        assert len(centers_indices) >= 1
        assert all(0 <= idx < len(X) for idx in centers_indices)


class TestMeanShiftModel:
    """Test MeanShiftModel class."""
    
    @pytest.fixture
    def model(self):
        """Create model instance."""
        return MeanShiftModel()
    
    @pytest.fixture
    def data(self):
        """Generate test data."""
        X, y = make_blobs(
            n_samples=50,
            centers=3,
            n_features=3,
            cluster_std=1.0,
            random_state=42
        )
        return X, y
    
    def test_initialization(self, model):
        """Test model initialization."""
        assert hasattr(model, 'model')
    
    def test_fit_predict(self, model, data):
        """Test fitting and prediction."""
        X, y = data
        
        model.fit(X)
        predictions = model.predict(X)
        
        assert len(predictions)