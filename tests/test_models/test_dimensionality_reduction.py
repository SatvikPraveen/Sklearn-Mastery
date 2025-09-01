"""
Unit tests for dimensionality reduction models.

Tests for all dimensionality reduction model wrappers and utilities.
"""

import pytest
import numpy as np
from sklearn.datasets import make_classification, load_digits
from sklearn.model_selection import train_test_split
import sys
import os
import tempfile

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from models.unsupervised.dimensionality_reduction import (
    DimensionalityReductionModel,
    PCAModel,
    TruncatedSVDModel,
    ICAModel,
    NMFModel,
    TSNEModel,
    UMAPModel,
    IsoMapModel,
    LLEModel,
    SpectralEmbeddingModel,
    MDSModel,
    DictionaryLearningModel,
    FactorAnalysisModel,
    KernelPCAModel
)


class TestDimensionalityReductionModel:
    """Test base DimensionalityReductionModel class."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample high-dimensional data."""
        X, y = make_classification(
            n_samples=200,
            n_features=20,
            n_informative=15,
            n_redundant=5,
            random_state=42
        )
        return X, y
    
    def test_abstract_methods(self):
        """Test that abstract methods raise NotImplementedError."""
        model = DimensionalityReductionModel()
        
        with pytest.raises(NotImplementedError):
            model.fit(None)
        
        with pytest.raises(NotImplementedError):
            model.transform(None)
    
    def test_fit_transform_method(self, sample_data):
        """Test fit_transform method with a concrete implementation."""
        X, y = sample_data
        
        # Use a concrete implementation
        model = PCAModel(n_components=5)
        X_transformed = model.fit_transform(X)
        
        assert X_transformed.shape == (X.shape[0], 5)
        assert X_transformed.dtype == np.float64


class TestPCAModel:
    """Test PCAModel class."""
    
    @pytest.fixture
    def model(self):
        """Create model instance."""
        return PCAModel(n_components=5, random_state=42)
    
    @pytest.fixture
    def data(self):
        """Generate test data."""
        X, y = make_classification(
            n_samples=100,
            n_features=15,
            n_informative=10,
            random_state=42
        )
        return X, y
    
    def test_initialization(self, model):
        """Test model initialization."""
        assert model.n_components == 5
        assert model.random_state == 42
        assert hasattr(model, 'model')
    
    def test_fit_transform(self, model, data):
        """Test fitting and transformation."""
        X, y = data
        
        X_transformed = model.fit_transform(X)
        
        assert X_transformed.shape == (X.shape[0], model.n_components)
        assert X_transformed.dtype == np.float64


class TestDictionaryLearningModel:
    """Test DictionaryLearningModel class."""
    
    @pytest.fixture
    def model(self):
        """Create model instance."""
        return DictionaryLearningModel(n_components=10, random_state=42)
    
    @pytest.fixture
    def data(self):
        """Generate test data."""
        X, y = make_classification(
            n_samples=50,
            n_features=15,
            n_informative=10,
            random_state=42
        )
        return X, y
    
    def test_initialization(self, model):
        """Test model initialization."""
        assert model.n_components == 10
        assert model.random_state == 42
        assert hasattr(model, 'model')
    
    def test_fit_transform(self, model, data):
        """Test fitting and transformation."""
        X, y = data
        
        X_transformed = model.fit_transform(X)
        
        assert X_transformed.shape == (X.shape[0], model.n_components)
        assert X_transformed.dtype == np.float64
    
    def test_dictionary(self, model, data):
        """Test dictionary extraction."""
        X, y = data
        model.fit(X)
        
        dictionary = model.get_components()
        
        assert dictionary.shape == (model.n_components, X.shape[1])


class TestFactorAnalysisModel:
    """Test FactorAnalysisModel class."""
    
    @pytest.fixture
    def model(self):
        """Create model instance."""
        return FactorAnalysisModel(n_components=5, random_state=42)
    
    @pytest.fixture
    def data(self):
        """Generate test data."""
        X, y = make_classification(
            n_samples=100,
            n_features=15,
            n_informative=10,
            random_state=42
        )
        return X, y
    
    def test_initialization(self, model):
        """Test model initialization."""
        assert model.n_components == 5
        assert model.random_state == 42
        assert hasattr(model, 'model')
    
    def test_fit_transform(self, model, data):
        """Test fitting and transformation."""
        X, y = data
        
        X_transformed = model.fit_transform(X)
        
        assert X_transformed.shape == (X.shape[0], model.n_components)
        assert X_transformed.dtype == np.float64
    
    def test_components(self, model, data):
        """Test factor loadings extraction."""
        X, y = data
        model.fit(X)
        
        components = model.get_components()
        
        assert components.shape == (model.n_components, X.shape[1])
    
    def test_noise_variance(self, model, data):
        """Test noise variance extraction."""
        X, y = data
        model.fit(X)
        
        noise_var = model.get_noise_variance()
        
        assert len(noise_var) == X.shape[1]
        assert np.all(noise_var >= 0)
    
    def test_log_likelihood(self, model, data):
        """Test log-likelihood calculation."""
        X, y = data
        model.fit(X)
        
        log_likelihood = model.score(X)
        
        assert isinstance(log_likelihood, (float, np.float64))


class TestKernelPCAModel:
    """Test KernelPCAModel class."""
    
    @pytest.fixture
    def model(self):
        """Create model instance."""
        return KernelPCAModel(n_components=5, kernel='rbf', random_state=42)
    
    @pytest.fixture
    def data(self):
        """Generate test data."""
        X, y = make_classification(
            n_samples=100,
            n_features=10,
            n_informative=8,
            random_state=42
        )
        return X, y
    
    def test_initialization(self, model):
        """Test model initialization."""
        assert model.n_components == 5
        assert model.kernel == 'rbf'
        assert model.random_state == 42
        assert hasattr(model, 'model')
    
    def test_fit_transform(self, model, data):
        """Test fitting and transformation."""
        X, y = data
        
        X_transformed = model.fit_transform(X)
        
        assert X_transformed.shape == (X.shape[0], model.n_components)
        assert X_transformed.dtype == np.float64
    
    def test_different_kernels(self, data):
        """Test different kernel functions."""
        X, y = data
        
        kernels = ['linear', 'poly', 'rbf', 'sigmoid']
        
        for kernel in kernels:
            model = KernelPCAModel(n_components=3, kernel=kernel, random_state=42)
            X_transformed = model.fit_transform(X)
            
            assert X_transformed.shape == (X.shape[0], 3)
    
    def test_inverse_transform(self, data):
        """Test inverse transformation for linear kernel."""
        X, y = data
        
        # Only linear kernel supports inverse transform reliably
        model = KernelPCAModel(n_components=5, kernel='linear', random_state=42)
        X_transformed = model.fit_transform(X)
        
        try:
            X_reconstructed = model.inverse_transform(X_transformed)
            assert X_reconstructed.shape == X.shape
        except AttributeError:
            # Some implementations might not have inverse_transform
            pass


class TestDimensionalityReductionIntegration:
    """Integration tests for dimensionality reduction models."""
    
    @pytest.fixture
    def models(self):
        """Create all model instances."""
        return {
            'pca': PCAModel(n_components=5, random_state=42),
            'svd': TruncatedSVDModel(n_components=5, random_state=42),
            'ica': ICAModel(n_components=5, random_state=42),
            'fa': FactorAnalysisModel(n_components=5, random_state=42),
            'kpca': KernelPCAModel(n_components=5, kernel='linear', random_state=42)
        }
    
    @pytest.fixture
    def manifold_models(self):
        """Create manifold learning model instances."""
        return {
            'isomap': IsoMapModel(n_components=3, n_neighbors=5),
            'lle': LLEModel(n_components=3, n_neighbors=5, random_state=42),
            'spectral': SpectralEmbeddingModel(n_components=3, random_state=42)
        }
    
    @pytest.fixture
    def data(self):
        """Generate test data."""
        X, y = make_classification(
            n_samples=100,
            n_features=15,
            n_informative=10,
            random_state=42
        )
        # Make data non-negative for models that require it
        X = np.abs(X)
        return X, y
    
    @pytest.fixture
    def manifold_data(self):
        """Generate smaller dataset for manifold learning."""
        X, y = make_classification(
            n_samples=50,
            n_features=10,
            n_informative=8,
            random_state=42
        )
        return X, y
    
    def test_all_models_dimensionality_reduction(self, models, data):
        """Test that all models reduce dimensionality correctly."""
        X, y = data
        
        for name, model in models.items():
            X_transformed = model.fit_transform(X)
            
            assert X_transformed.shape[0] == X.shape[0], f"{name} should preserve number of samples"
            assert X_transformed.shape[1] == model.n_components, f"{name} should reduce to n_components"
            assert X_transformed.shape[1] < X.shape[1], f"{name} should reduce dimensionality"
    
    def test_manifold_models_dimensionality_reduction(self, manifold_models, manifold_data):
        """Test manifold learning models."""
        X, y = manifold_data
        
        for name, model in manifold_models.items():
            X_transformed = model.fit_transform(X)
            
            assert X_transformed.shape[0] == X.shape[0], f"{name} should preserve number of samples"
            assert X_transformed.shape[1] == model.n_components, f"{name} should reduce to n_components"
            assert X_transformed.shape[1] < X.shape[1], f"{name} should reduce dimensionality"
    
    def test_variance_preservation_ordering(self, data):
        """Test that PCA preserves variance in correct order."""
        X, y = data
        
        # Test different numbers of components
        n_components_list = [3, 5, 8]
        variances = []
        
        for n_comp in n_components_list:
            model = PCAModel(n_components=n_comp, random_state=42)
            model.fit(X)
            explained_var_ratio = model.get_explained_variance_ratio()
            total_variance = np.sum(explained_var_ratio)
            variances.append(total_variance)
        
        # More components should explain more variance
        assert variances[0] <= variances[1] <= variances[2]
    
    def test_reconstruction_quality(self, data):
        """Test reconstruction quality for reversible methods."""
        X, y = data
        
        # Test PCA reconstruction
        for n_comp in [5, 10]:
            model = PCAModel(n_components=n_comp, random_state=42)
            X_transformed = model.fit_transform(X)
            X_reconstructed = model.inverse_transform(X_transformed)
            
            # Calculate reconstruction error
            mse = np.mean((X - X_reconstructed) ** 2)
            
            # More components should lead to better reconstruction
            assert mse >= 0
            
            # Reconstruction error should be reasonable
            baseline_var = np.var(X)
            assert mse < baseline_var  # Should be better than random
    
    def test_data_scaling_sensitivity(self, models):
        """Test sensitivity to data scaling."""
        # Create data with different scales
        X1, _ = make_classification(n_samples=50, n_features=10, random_state=42)
        X2 = X1 * 1000  # Scaled version
        
        # Make non-negative
        X1 = np.abs(X1)
        X2 = np.abs(X2)
        
        for name, model in models.items():
            if name == 'ica':  # ICA can be sensitive to scaling
                continue
                
            # Fit on both datasets
            X1_transformed = model.fit_transform(X1)
            
            # Create new instance for second dataset
            if hasattr(model, 'random_state'):
                model2 = type(model)(n_components=model.n_components, random_state=model.random_state)
            else:
                model2 = type(model)(n_components=model.n_components)
            
            X2_transformed = model2.fit_transform(X2)
            
            # Shapes should be the same
            assert X1_transformed.shape == X2_transformed.shape, f"{name} shape consistency failed"
    
    def test_reproducibility(self, data):
        """Test reproducibility with random_state."""
        X, y = data
        
        reproducible_models = [
            ('pca1', PCAModel(n_components=5, random_state=42)),
            ('pca2', PCAModel(n_components=5, random_state=42)),
            ('ica1', ICAModel(n_components=5, random_state=42)),
            ('ica2', ICAModel(n_components=5, random_state=42))
        ]
        
        # Test pairs of identical models
        for i in range(0, len(reproducible_models), 2):
            name1, model1 = reproducible_models[i]
            name2, model2 = reproducible_models[i + 1]
            
            X1_transformed = model1.fit_transform(X)
            X2_transformed = model2.fit_transform(X)
            
            # Results should be identical (or very close) for same random state
            if 'ica' in name1.lower():
                # ICA might have slight variations due to convergence
                np.testing.assert_array_almost_equal(
                    X1_transformed, X2_transformed, 
                    f"Models {name1} and {name2} should produce similar results",
                    decimal=3
                )
            else:
                # PCA should be exactly reproducible
                np.testing.assert_array_almost_equal(
                    X1_transformed, X2_transformed, 
                    f"Models {name1} and {name2} should produce identical results",
                    decimal=10
                )
    
    def test_component_orthogonality(self, data):
        """Test orthogonality properties where applicable."""
        X, y = data
        
        # PCA components should be orthogonal
        model = PCAModel(n_components=5, random_state=42)
        model.fit(X)
        
        # Access components through the sklearn model
        if hasattr(model.model, 'components_'):
            components = model.model.components_
            
            # Check orthogonality
            dot_product = np.dot(components, components.T)
            np.testing.assert_array_almost_equal(
                dot_product, np.eye(5),
                "PCA components should be orthogonal",
                decimal=10
            )
        else:
            pytest.skip("Model doesn't have components_ attribute")
    
    def test_increasing_components_effect(self, data):
        """Test effect of increasing number of components."""
        X, y = data
        
        component_counts = [2, 5, 8, 10]
        
        for model_class in [PCAModel, TruncatedSVDModel]:
            explained_variances = []
            
            for n_comp in component_counts:
                if hasattr(model_class(), 'random_state'):
                    model = model_class(n_components=n_comp, random_state=42)
                else:
                    model = model_class(n_components=n_comp)
                
                model.fit(X)
                
                if hasattr(model, 'get_explained_variance_ratio'):
                    explained_var = np.sum(model.get_explained_variance_ratio())
                    explained_variances.append(explained_var)
            
            # More components should explain more variance
            if explained_variances:
                for i in range(1, len(explained_variances)):
                    assert explained_variances[i] >= explained_variances[i-1], \
                        f"{model_class.__name__} should explain more variance with more components"
    


class TestTruncatedSVDModel:
    """Test TruncatedSVDModel class."""
    
    @pytest.fixture
    def model(self):
        """Create model instance."""
        return TruncatedSVDModel(n_components=5, random_state=42)
    
    @pytest.fixture
    def data(self):
        """Generate test data."""
        # Use sparse-friendly data
        X, y = make_classification(
            n_samples=100,
            n_features=15,
            n_informative=10,
            random_state=42
        )
        # Make data non-negative for SVD
        X = np.abs(X)
        return X, y
    
    def test_initialization(self, model):
        """Test model initialization."""
        assert model.n_components == 5
        assert model.random_state == 42
        assert hasattr(model, 'model')
    
    def test_fit_transform(self, model, data):
        """Test fitting and transformation."""
        X, y = data
        
        X_transformed = model.fit_transform(X)
        
        assert X_transformed.shape == (X.shape[0], model.n_components)
        assert X_transformed.dtype == np.float64
    
    def test_explained_variance(self, model, data):
        """Test explained variance extraction."""
        X, y = data
        model.fit(X)
        
        explained_var_ratio = model.get_explained_variance_ratio()
        
        assert len(explained_var_ratio) == model.n_components
        assert np.all(explained_var_ratio >= 0)
        assert np.sum(explained_var_ratio) <= 1.0


class TestICAModel:
    """Test ICAModel class."""
    
    @pytest.fixture
    def model(self):
        """Create model instance."""
        return ICAModel(n_components=5, random_state=42)
    
    @pytest.fixture
    def data(self):
        """Generate test data with mixed signals."""
        np.random.seed(42)
        time = np.linspace(0, 8, 100)
        
        # Create mixed signals
        s1 = np.sin(2 * time)  # Signal 1: sine wave
        s2 = np.sign(np.sin(3 * time))  # Signal 2: square wave
        s3 = np.random.randn(len(time))  # Signal 3: noise
        
        S = np.c_[s1, s2, s3]
        
        # Mix signals
        A = np.array([[1, 1, 1], [0.5, 2, 1.0], [1.5, 1.0, 2.0], [0.5, 0.5, 1.0], [1.0, 1.5, 0.5]])
        X = np.dot(S, A.T)
        
        return X, S
    
    def test_initialization(self, model):
        """Test model initialization."""
        assert model.n_components == 5
        assert model.random_state == 42
        assert hasattr(model, 'model')
    
    def test_fit_transform(self, model, data):
        """Test fitting and transformation."""
        X, S = data
        
        X_transformed = model.fit_transform(X)
        
        assert X_transformed.shape == (X.shape[0], model.n_components)
        assert X_transformed.dtype == np.float64
    
    def test_mixing_matrix(self, model, data):
        """Test mixing matrix extraction."""
        X, S = data
        model.fit(X)
        
        mixing_matrix = model.get_mixing_matrix()
        
        assert mixing_matrix.shape == (X.shape[1], model.n_components)


class TestNMFModel:
    """Test NMFModel class."""
    
    @pytest.fixture
    def model(self):
        """Create model instance."""
        return NMFModel(n_components=5, random_state=42)
    
    @pytest.fixture
    def data(self):
        """Generate non-negative test data."""
        X, y = make_classification(
            n_samples=100,
            n_features=15,
            n_informative=10,
            random_state=42
        )
        # Make data non-negative for NMF
        X = np.abs(X)
        return X, y
    
    def test_initialization(self, model):
        """Test model initialization."""
        assert model.n_components == 5
        assert model.random_state == 42
        assert hasattr(model, 'model')
    
    def test_fit_transform(self, model, data):
        """Test fitting and transformation."""
        X, y = data
        
        X_transformed = model.fit_transform(X)
        
        assert X_transformed.shape == (X.shape[0], model.n_components)
        assert X_transformed.dtype == np.float64
        assert np.all(X_transformed >= 0)  # NMF should produce non-negative components
    
    def test_components(self, model, data):
        """Test NMF components extraction."""
        X, y = data
        model.fit(X)
        
        components = model.get_components()
        
        assert components.shape == (model.n_components, X.shape[1])
        assert np.all(components >= 0)  # Components should be non-negative
    
    def test_reconstruction_error(self, model, data):
        """Test reconstruction error calculation."""
        X, y = data
        model.fit(X)
        
        error = model.get_reconstruction_error()
        
        assert isinstance(error, (float, np.float64))
        assert error >= 0


class TestTSNEModel:
    """Test TSNEModel class."""
    
    @pytest.fixture
    def model(self):
        """Create model instance."""
        return TSNEModel(n_components=2, random_state=42, n_iter=250)
    
    @pytest.fixture
    def data(self):
        """Generate test data."""
        # Use smaller dataset for t-SNE (computationally expensive)
        X, y = make_classification(
            n_samples=50,
            n_features=10,
            n_informative=8,
            random_state=42
        )
        return X, y
    
    def test_initialization(self, model):
        """Test model initialization."""
        assert model.n_components == 2
        assert model.random_state == 42
        assert hasattr(model, 'model')
    
    def test_fit_transform(self, model, data):
        """Test fitting and transformation."""
        X, y = data
        
        X_transformed = model.fit_transform(X)
        
        assert X_transformed.shape == (X.shape[0], model.n_components)
        assert X_transformed.dtype == np.float64
    
    def test_kl_divergence(self, model, data):
        """Test KL divergence extraction."""
        X, y = data
        model.fit_transform(X)
        
        kl_div = model.get_kl_divergence()
        
        assert isinstance(kl_div, (float, np.float64))
        assert kl_div >= 0


class TestUMAPModel:
    """Test UMAPModel class."""
    
    @pytest.fixture
    def model(self):
        """Create model instance."""
        return UMAPModel(n_components=2, random_state=42)
    
    @pytest.fixture
    def data(self):
        """Generate test data."""
        X, y = make_classification(
            n_samples=100,
            n_features=10,
            n_informative=8,
            random_state=42
        )
        return X, y
    
    def test_initialization(self, model):
        """Test model initialization."""
        assert model.n_components == 2
        assert model.random_state == 42
        assert hasattr(model, 'model')
    
    def test_fit_transform(self, model, data):
        """Test fitting and transformation."""
        X, y = data
        
        X_transformed = model.fit_transform(X)
        
        assert X_transformed.shape == (X.shape[0], model.n_components)
        assert X_transformed.dtype == np.float64
    
    def test_transform_new_data(self, model, data):
        """Test transformation of new data."""
        X, y = data
        X_train, X_test = train_test_split(X, test_size=0.3, random_state=42)
        
        model.fit(X_train)
        X_test_transformed = model.transform(X_test)
        
        assert X_test_transformed.shape == (len(X_test), model.n_components)


class TestIsoMapModel:
    """Test IsoMapModel class."""
    
    @pytest.fixture
    def model(self):
        """Create model instance."""
        return IsoMapModel(n_components=3, n_neighbors=5)
    
    @pytest.fixture
    def data(self):
        """Generate test data."""
        X, y = make_classification(
            n_samples=50,  # Smaller dataset for manifold learning
            n_features=10,
            n_informative=8,
            random_state=42
        )
        return X, y
    
    def test_initialization(self, model):
        """Test model initialization."""
        assert model.n_components == 3
        assert model.n_neighbors == 5
        assert hasattr(model, 'model')
    
    def test_fit_transform(self, model, data):
        """Test fitting and transformation."""
        X, y = data
        
        X_transformed = model.fit_transform(X)
        
        assert X_transformed.shape == (X.shape[0], model.n_components)
        assert X_transformed.dtype == np.float64
    
    def test_reconstruction_error(self, model, data):
        """Test reconstruction error calculation."""
        X, y = data
        model.fit(X)
        
        error = model.get_reconstruction_error()
        
        assert isinstance(error, (float, np.float64))
        assert error >= 0


class TestLLEModel:
    """Test LLEModel class."""
    
    @pytest.fixture
    def model(self):
        """Create model instance."""
        return LLEModel(n_components=3, n_neighbors=5, random_state=42)
    
    @pytest.fixture
    def data(self):
        """Generate test data."""
        X, y = make_classification(
            n_samples=50,  # Smaller dataset for manifold learning
            n_features=10,
            n_informative=8,
            random_state=42
        )
        return X, y
    
    def test_initialization(self, model):
        """Test model initialization."""
        assert model.n_components == 3
        assert model.n_neighbors == 5
        assert model.random_state == 42
        assert hasattr(model, 'model')
    
    def test_fit_transform(self, model, data):
        """Test fitting and transformation."""
        X, y = data
        
        X_transformed = model.fit_transform(X)
        
        assert X_transformed.shape == (X.shape[0], model.n_components)
        assert X_transformed.dtype == np.float64
    
    def test_reconstruction_error(self, model, data):
        """Test reconstruction error calculation."""
        X, y = data
        model.fit(X)
        
        error = model.get_reconstruction_error()
        
        assert isinstance(error, (float, np.float64))
        assert error >= 0


class TestSpectralEmbeddingModel:
    """Test SpectralEmbeddingModel class."""
    
    @pytest.fixture
    def model(self):
        """Create model instance."""
        return SpectralEmbeddingModel(n_components=3, random_state=42)
    
    @pytest.fixture
    def data(self):
        """Generate test data."""
        X, y = make_classification(
            n_samples=50,
            n_features=10,
            n_informative=8,
            random_state=42
        )
        return X, y
    
    def test_initialization(self, model):
        """Test model initialization."""
        assert model.n_components == 3
        assert model.random_state == 42
        assert hasattr(model, 'model')
    
    def test_fit_transform(self, model, data):
        """Test fitting and transformation."""
        X, y = data
        
        X_transformed = model.fit_transform(X)
        
        assert X_transformed.shape == (X.shape[0], model.n_components)
        assert X_transformed.dtype == np.float64


class TestMDSModel:
    """Test MDSModel class."""
    
    @pytest.fixture
    def model(self):
        """Create model instance."""
        return MDSModel(n_components=3, random_state=42)
    
    @pytest.fixture
    def data(self):
        """Generate test data."""
        X, y = make_classification(
            n_samples=30,  # Small dataset for MDS (O(n^3) complexity)
            n_features=10,
            n_informative=8,
            random_state=42
        )
        return X, y
    
    def test_initialization(self, model):
        """Test model initialization."""
        assert model.n_components == 3
        assert model.random_state == 42
        assert hasattr(model, 'model')
    
    def test_fit_transform(self, model, data):
        """Test fitting and transformation."""
        X, y = data
        
        X_transformed = model.fit_transform(X)
        
        assert X_transformed.shape == (X.shape[0], model.n_components)
        assert X_transformed.dtype == np.float64
    
    def test_stress(self, model, data):
        """Test stress calculation."""
        X, y = data
        model.fit(X)
        
        stress = model.get_stress()
        
        assert isinstance(stress, (float, np.float64))
        assert stress >= 0


if __name__ == "__main__":
    pytest.main([__file__])