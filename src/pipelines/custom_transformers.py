"""Custom transformers for advanced pipeline construction."""

import numpy as np
import pandas as pd
from typing import Union, List, Optional, Dict, Any, Tuple
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer
import warnings

from ..config.logging_config import LoggerMixin


class OutlierRemover(BaseEstimator, TransformerMixin, LoggerMixin):
    """Custom transformer for outlier detection and removal."""
    
    def __init__(
        self, 
        method: str = 'isolation_forest',
        contamination: float = 0.1,
        n_neighbors: int = 20,
        random_state: int = 42
    ):
        """Initialize outlier remover.
        
        Args:
            method: Method for outlier detection ('isolation_forest', 'lof', 'z_score').
            contamination: Expected proportion of outliers.
            n_neighbors: Number of neighbors for LOF.
            random_state: Random state for reproducibility.
        """
        self.method = method
        self.contamination = contamination
        self.n_neighbors = n_neighbors
        self.random_state = random_state
        self.outlier_detector_ = None
        self.outlier_mask_ = None
        
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """Fit the outlier detector.
        
        Args:
            X: Feature matrix.
            y: Target vector (ignored).
            
        Returns:
            self: Fitted transformer.
        """
        X = self._validate_input(X)
        
        if self.method == 'isolation_forest':
            self.outlier_detector_ = IsolationForest(
                contamination=self.contamination,
                random_state=self.random_state,
                n_jobs=-1
            )
            outlier_labels = self.outlier_detector_.fit_predict(X)
            
        elif self.method == 'lof':
            self.outlier_detector_ = LocalOutlierFactor(
                n_neighbors=self.n_neighbors,
                contamination=self.contamination,
                n_jobs=-1
            )
            outlier_labels = self.outlier_detector_.fit_predict(X)
            
        elif self.method == 'z_score':
            # Z-score based outlier detection
            z_scores = np.abs((X - np.mean(X, axis=0)) / np.std(X, axis=0))
            # Mark as outlier if any feature has z-score > 3
            outlier_labels = np.where(np.any(z_scores > 3, axis=1), -1, 1)
            
        else:
            raise ValueError(f"Unknown outlier detection method: {self.method}")
        
        # Store inlier mask (outlier_labels: 1 = inlier, -1 = outlier)
        self.outlier_mask_ = outlier_labels == 1
        
        n_outliers = np.sum(~self.outlier_mask_)
        self.logger.info(f"Detected {n_outliers} outliers using {self.method}")
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Remove outliers from the data.
        
        Args:
            X: Feature matrix.
            
        Returns:
            X_clean: Feature matrix with outliers removed.
        """
        if self.outlier_mask_ is None:
            raise ValueError("Transformer must be fitted before transform.")
            
        X = self._validate_input(X)
        
        # During training, remove detected outliers
        if len(X) == len(self.outlier_mask_):
            return X[self.outlier_mask_]
        
        # During inference, keep all data (outliers already detected during fit)
        return X
    
    def _validate_input(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Validate and convert input to numpy array."""
        if isinstance(X, pd.DataFrame):
            return X.values
        return np.asarray(X)


class FeatureInteractionCreator(BaseEstimator, TransformerMixin, LoggerMixin):
    """Creates polynomial and interaction features intelligently."""
    
    def __init__(
        self,
        degree: int = 2,
        interaction_only: bool = False,
        include_bias: bool = False,
        max_features: Optional[int] = None
    ):
        """Initialize feature interaction creator.
        
        Args:
            degree: Maximum degree of polynomial features.
            interaction_only: If True, only interaction features are produced.
            include_bias: If True, include bias column.
            max_features: Maximum number of features to keep after creation.
        """
        self.degree = degree
        self.interaction_only = interaction_only
        self.include_bias = include_bias
        self.max_features = max_features
        self.feature_names_ = None
        self.selected_features_ = None
        
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """Fit the feature interaction creator.
        
        Args:
            X: Feature matrix.
            y: Target vector (used for feature selection).
            
        Returns:
            self: Fitted transformer.
        """
        X = self._validate_input(X)
        n_samples, n_features = X.shape
        
        # Create polynomial features
        X_poly = self._create_polynomial_features(X)
        
        # Feature selection if max_features is specified
        if self.max_features is not None and X_poly.shape[1] > self.max_features:
            if y is not None:
                self.logger.info(f"Selecting {self.max_features} best interaction features")
                selector = SelectKBest(
                    score_func=f_regression if self._is_regression(y) else f_classif,
                    k=self.max_features
                )
                selector.fit(X_poly, y)
                self.selected_features_ = selector.get_support()
            else:
                # Random selection if no target provided
                self.selected_features_ = np.zeros(X_poly.shape[1], dtype=bool)
                selected_indices = np.random.choice(
                    X_poly.shape[1], 
                    size=min(self.max_features, X_poly.shape[1]), 
                    replace=False
                )
                self.selected_features_[selected_indices] = True
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Create polynomial and interaction features.
        
        Args:
            X: Feature matrix.
            
        Returns:
            X_poly: Matrix with polynomial/interaction features.
        """
        X = self._validate_input(X)
        X_poly = self._create_polynomial_features(X)
        
        # Apply feature selection if fitted
        if self.selected_features_ is not None:
            X_poly = X_poly[:, self.selected_features_]
        
        return X_poly
    
    def _create_polynomial_features(self, X: np.ndarray) -> np.ndarray:
        """Create polynomial features manually for better control."""
        n_samples, n_features = X.shape
        features = [X]  # Start with original features
        
        if self.include_bias:
            bias = np.ones((n_samples, 1))
            features.append(bias)
        
        # Create interaction features
        if self.degree >= 2:
            for i in range(n_features):
                for j in range(i + (1 if self.interaction_only else 0), n_features):
                    interaction = (X[:, i] * X[:, j]).reshape(-1, 1)
                    features.append(interaction)
        
        # Create higher-degree features if needed
        if self.degree >= 3 and not self.interaction_only:
            for i in range(n_features):
                cubic = (X[:, i] ** 3).reshape(-1, 1)
                features.append(cubic)
        
        return np.hstack(features)
    
    def _is_regression(self, y: np.ndarray) -> bool:
        """Check if target is continuous (regression) or discrete (classification)."""
        return len(np.unique(y)) > 10 or y.dtype.kind == 'f'
    
    def _validate_input(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Validate and convert input to numpy array."""
        if isinstance(X, pd.DataFrame):
            return X.select_dtypes(include=[np.number]).values
        return np.asarray(X)


class DomainSpecificEncoder(BaseEstimator, TransformerMixin, LoggerMixin):
    """Custom encoding strategies for different data domains."""
    
    def __init__(
        self,
        categorical_strategy: str = 'onehot',
        ordinal_strategy: str = 'ordinal',
        high_cardinality_threshold: int = 10,
        rare_category_threshold: float = 0.01
    ):
        """Initialize domain-specific encoder.
        
        Args:
            categorical_strategy: Strategy for categorical encoding ('onehot', 'target', 'binary').
            ordinal_strategy: Strategy for ordinal encoding ('ordinal', 'label').
            high_cardinality_threshold: Threshold for considering a feature high cardinality.
            rare_category_threshold: Threshold for grouping rare categories.
        """
        self.categorical_strategy = categorical_strategy
        self.ordinal_strategy = ordinal_strategy
        self.high_cardinality_threshold = high_cardinality_threshold
        self.rare_category_threshold = rare_category_threshold
        self.encoders_ = {}
        self.feature_names_ = []
        self.categorical_features_ = []
        self.numerical_features_ = []
        
    def fit(self, X: pd.DataFrame, y: Optional[np.ndarray] = None):
        """Fit the domain-specific encoder.
        
        Args:
            X: Feature matrix (must be DataFrame).
            y: Target vector.
            
        Returns:
            self: Fitted transformer.
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("DomainSpecificEncoder requires pandas DataFrame")
        
        self.categorical_features_ = []
        self.numerical_features_ = []
        
        for col in X.columns:
            if X[col].dtype.name == 'category' or X[col].dtype == 'object':
                self.categorical_features_.append(col)
            else:
                self.numerical_features_.append(col)
        
        self.logger.info(f"Found {len(self.categorical_features_)} categorical and {len(self.numerical_features_)} numerical features")
        
        # Fit encoders for categorical features
        for col in self.categorical_features_:
            self._fit_categorical_encoder(X[col], col, y)
        
        return self
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform the data using fitted encoders.
        
        Args:
            X: Feature matrix.
            
        Returns:
            X_encoded: Encoded feature matrix.
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("DomainSpecificEncoder requires pandas DataFrame")
        
        encoded_features = []
        
        # Transform numerical features (keep as-is)
        if self.numerical_features_:
            encoded_features.append(X[self.numerical_features_].values)
        
        # Transform categorical features
        for col in self.categorical_features_:
            encoded_col = self._transform_categorical_feature(X[col], col)
            if encoded_col.ndim == 1:
                encoded_col = encoded_col.reshape(-1, 1)
            encoded_features.append(encoded_col)
        
        if encoded_features:
            return np.hstack(encoded_features)
        else:
            return np.empty((len(X), 0))
    
    def _fit_categorical_encoder(self, series: pd.Series, col_name: str, y: Optional[np.ndarray]):
        """Fit encoder for a single categorical feature."""
        unique_values = series.nunique()
        
        if unique_values <= self.high_cardinality_threshold:
            if self.categorical_strategy == 'onehot':
                from sklearn.preprocessing import OneHotEncoder
                encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                encoder.fit(series.values.reshape(-1, 1))
                self.encoders_[col_name] = encoder
                
            elif self.categorical_strategy == 'target' and y is not None:
                # Target encoding
                target_means = series.groupby(series).apply(
                    lambda x: y[x.index].mean()
                ).to_dict()
                self.encoders_[col_name] = target_means
                
            else:  # Label encoding as fallback
                from sklearn.preprocessing import LabelEncoder
                encoder = LabelEncoder()
                encoder.fit(series.values)
                self.encoders_[col_name] = encoder
        else:
            # High cardinality - use target encoding or hashing
            if y is not None:
                target_means = series.groupby(series).apply(
                    lambda x: y[x.index].mean()
                ).to_dict()
                self.encoders_[col_name] = target_means
            else:
                # Use hash encoding for unsupervised case
                from sklearn.feature_extraction import FeatureHasher
                hasher = FeatureHasher(n_features=min(32, unique_values), input_type='string')
                self.encoders_[col_name] = hasher
    
    def _transform_categorical_feature(self, series: pd.Series, col_name: str) -> np.ndarray:
        """Transform a single categorical feature."""
        encoder = self.encoders_[col_name]
        
        if hasattr(encoder, 'transform'):  # sklearn encoder
            if hasattr(encoder, 'n_features_'):  # FeatureHasher
                return encoder.transform(series.astype(str).values.reshape(-1, 1)).toarray()
            else:  # OneHotEncoder or LabelEncoder
                return encoder.transform(series.values.reshape(-1, 1))
        else:  # Target encoding (dictionary)
            return np.array([encoder.get(val, encoder.get('unknown', 0)) for val in series.values])


class AdvancedImputer(BaseEstimator, TransformerMixin, LoggerMixin):
    """Advanced imputation with multiple strategies."""
    
    def __init__(
        self,
        numerical_strategy: str = 'knn',
        categorical_strategy: str = 'mode',
        n_neighbors: int = 5
    ):
        """Initialize advanced imputer.
        
        Args:
            numerical_strategy: Strategy for numerical features ('mean', 'median', 'knn', 'iterative').
            categorical_strategy: Strategy for categorical features ('mode', 'constant').
            n_neighbors: Number of neighbors for KNN imputation.
        """
        self.numerical_strategy = numerical_strategy
        self.categorical_strategy = categorical_strategy
        self.n_neighbors = n_neighbors
        self.numerical_imputer_ = None
        self.categorical_imputer_ = None
        self.numerical_features_ = None
        self.categorical_features_ = None
        
    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Optional[np.ndarray] = None):
        """Fit the imputer.
        
        Args:
            X: Feature matrix.
            y: Target vector (ignored).
            
        Returns:
            self: Fitted transformer.
        """
        if isinstance(X, pd.DataFrame):
            self.numerical_features_ = X.select_dtypes(include=[np.number]).columns.tolist()
            self.categorical_features_ = X.select_dtypes(include=['object', 'category']).columns.tolist()
            
            # Fit numerical imputer
            if self.numerical_features_:
                X_num = X[self.numerical_features_].values
                self._fit_numerical_imputer(X_num)
                
            # Fit categorical imputer
            if self.categorical_features_:
                X_cat = X[self.categorical_features_]
                self._fit_categorical_imputer(X_cat)
        else:
            # Assume all numerical for numpy arrays
            self._fit_numerical_imputer(X)
            
        return self
    
    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
        """Transform the data using fitted imputers.
        
        Args:
            X: Feature matrix.
            
        Returns:
            X_imputed: Imputed feature matrix.
        """
        if isinstance(X, pd.DataFrame):
            X_imputed = X.copy()
            
            # Transform numerical features
            if self.numerical_features_ and self.numerical_imputer_:
                X_num_imputed = self.numerical_imputer_.transform(X[self.numerical_features_].values)
                X_imputed[self.numerical_features_] = X_num_imputed
                
            # Transform categorical features
            if self.categorical_features_ and self.categorical_imputer_:
                for col in self.categorical_features_:
                    X_imputed[col] = X_imputed[col].fillna(self.categorical_imputer_[col])
                    
            return X_imputed
        else:
            # Numpy array - assume numerical
            if self.numerical_imputer_:
                return self.numerical_imputer_.transform(X)
            return X
    
    def _fit_numerical_imputer(self, X_num: np.ndarray):
        """Fit numerical imputer."""
        if self.numerical_strategy == 'knn':
            self.numerical_imputer_ = KNNImputer(n_neighbors=self.n_neighbors)
        elif self.numerical_strategy == 'iterative':
            from sklearn.experimental import enable_iterative_imputer
            from sklearn.impute import IterativeImputer
            self.numerical_imputer_ = IterativeImputer(random_state=42)
        else:
            from sklearn.impute import SimpleImputer
            self.numerical_imputer_ = SimpleImputer(strategy=self.numerical_strategy)
            
        self.numerical_imputer_.fit(X_num)
    
    def _fit_categorical_imputer(self, X_cat: pd.DataFrame):
        """Fit categorical imputer."""
        self.categorical_imputer_ = {}
        
        for col in X_cat.columns:
            if self.categorical_strategy == 'mode':
                mode_value = X_cat[col].mode()
                self.categorical_imputer_[col] = mode_value[0] if len(mode_value) > 0 else 'unknown'
            else:  # constant
                self.categorical_imputer_[col] = 'missing'


class FeatureScaler(BaseEstimator, TransformerMixin, LoggerMixin):
    """Intelligent feature scaling based on data distribution."""
    
    def __init__(
        self,
        strategy: str = 'auto',
        robust_threshold: float = 0.1
    ):
        """Initialize feature scaler.
        
        Args:
            strategy: Scaling strategy ('auto', 'standard', 'robust', 'minmax', 'quantile').
            robust_threshold: Threshold for outlier detection to choose robust scaling.
        """
        self.strategy = strategy
        self.robust_threshold = robust_threshold
        self.scalers_ = {}
        self.feature_strategies_ = {}
        
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """Fit the feature scalers.
        
        Args:
            X: Feature matrix.
            y: Target vector (ignored).
            
        Returns:
            self: Fitted transformer.
        """
        X = self._validate_input(X)
        n_features = X.shape[1]
        
        for i in range(n_features):
            feature = X[:, i]
            
            if self.strategy == 'auto':
                # Automatically choose scaling strategy based on data distribution
                strategy = self._choose_scaling_strategy(feature)
            else:
                strategy = self.strategy
                
            self.feature_strategies_[i] = strategy
            
            # Fit appropriate scaler
            if strategy == 'standard':
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
            elif strategy == 'robust':
                from sklearn.preprocessing import RobustScaler
                scaler = RobustScaler()
            elif strategy == 'minmax':
                from sklearn.preprocessing import MinMaxScaler
                scaler = MinMaxScaler()
            elif strategy == 'quantile':
                from sklearn.preprocessing import QuantileTransformer
                scaler = QuantileTransformer(n_quantiles=min(1000, len(feature)))
            else:
                continue  # No scaling
                
            scaler.fit(feature.reshape(-1, 1))
            self.scalers_[i] = scaler
            
        self.logger.info(f"Fitted scalers with strategies: {self.feature_strategies_}")
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform features using fitted scalers.
        
        Args:
            X: Feature matrix.
            
        Returns:
            X_scaled: Scaled feature matrix.
        """
        X = self._validate_input(X)
        X_scaled = X.copy()
        
        for i, scaler in self.scalers_.items():
            X_scaled[:, i] = scaler.transform(X[:, i].reshape(-1, 1)).ravel()
            
        return X_scaled
    
    def _choose_scaling_strategy(self, feature: np.ndarray) -> str:
        """Choose appropriate scaling strategy based on feature distribution."""
        # Check for outliers using IQR method
        Q1, Q3 = np.percentile(feature, [25, 75])
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outlier_fraction = np.mean((feature < lower_bound) | (feature > upper_bound))
        
        if outlier_fraction > self.robust_threshold:
            return 'robust'
        
        # Check if data is already in [0, 1] range
        if np.min(feature) >= 0 and np.max(feature) <= 1:
            return 'none'
        
        # Check for highly skewed data
        from scipy import stats
        try:
            skewness = np.abs(stats.skew(feature))
            if skewness > 2:
                return 'quantile'
        except:
            pass
        
        # Default to standard scaling
        return 'standard'
    
    def _validate_input(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Validate and convert input to numpy array."""
        if isinstance(X, pd.DataFrame):
            return X.select_dtypes(include=[np.number]).values
        return np.asarray(X)


class TimeSeriesFeatureCreator(BaseEstimator, TransformerMixin, LoggerMixin):
    """Create time series features from datetime columns."""
    
    def __init__(
        self,
        datetime_column: str = 'date',
        create_lags: bool = True,
        lag_periods: List[int] = None,
        create_rolling: bool = True,
        rolling_windows: List[int] = None,
        create_cyclical: bool = True
    ):
        """Initialize time series feature creator.
        
        Args:
            datetime_column: Name of datetime column.
            create_lags: Whether to create lag features.
            lag_periods: List of lag periods to create.
            create_rolling: Whether to create rolling statistics.
            rolling_windows: List of rolling window sizes.
            create_cyclical: Whether to create cyclical features (sin/cos).
        """
        self.datetime_column = datetime_column
        self.create_lags = create_lags
        self.lag_periods = lag_periods or [1, 7, 30]
        self.create_rolling = create_rolling
        self.rolling_windows = rolling_windows or [7, 30]
        self.create_cyclical = create_cyclical
        self.feature_names_ = []
        
    def fit(self, X: pd.DataFrame, y: Optional[np.ndarray] = None):
        """Fit the time series feature creator.
        
        Args:
            X: Feature matrix with datetime column.
            y: Target vector (ignored).
            
        Returns:
            self: Fitted transformer.
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("TimeSeriesFeatureCreator requires pandas DataFrame")
            
        if self.datetime_column not in X.columns:
            raise ValueError(f"Datetime column '{self.datetime_column}' not found")
            
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data by creating time series features.
        
        Args:
            X: Feature matrix.
            
        Returns:
            X_transformed: DataFrame with additional time series features.
        """
        X_transformed = X.copy()
        
        # Ensure datetime column is datetime type
        if not pd.api.types.is_datetime64_any_dtype(X_transformed[self.datetime_column]):
            X_transformed[self.datetime_column] = pd.to_datetime(X_transformed[self.datetime_column])
        
        dt_col = X_transformed[self.datetime_column]
        
        # Basic datetime features
        X_transformed['year'] = dt_col.dt.year
        X_transformed['month'] = dt_col.dt.month
        X_transformed['day'] = dt_col.dt.day
        X_transformed['dayofweek'] = dt_col.dt.dayofweek
        X_transformed['dayofyear'] = dt_col.dt.dayofyear
        X_transformed['quarter'] = dt_col.dt.quarter
        X_transformed['is_weekend'] = dt_col.dt.dayofweek >= 5
        X_transformed['is_month_start'] = dt_col.dt.is_month_start
        X_transformed['is_month_end'] = dt_col.dt.is_month_end
        
        # Cyclical features
        if self.create_cyclical:
            X_transformed['month_sin'] = np.sin(2 * np.pi * X_transformed['month'] / 12)
            X_transformed['month_cos'] = np.cos(2 * np.pi * X_transformed['month'] / 12)
            X_transformed['dayofweek_sin'] = np.sin(2 * np.pi * X_transformed['dayofweek'] / 7)
            X_transformed['dayofweek_cos'] = np.cos(2 * np.pi * X_transformed['dayofweek'] / 7)
            X_transformed['dayofyear_sin'] = np.sin(2 * np.pi * X_transformed['dayofyear'] / 365.25)
            X_transformed['dayofyear_cos'] = np.cos(2 * np.pi * X_transformed['dayofyear'] / 365.25)
        
        # Time since epoch (useful for trend)
        X_transformed['days_since_epoch'] = (dt_col - pd.Timestamp('1970-01-01')).dt.days
        
        return X_transformed


class TextFeatureExtractor(BaseEstimator, TransformerMixin, LoggerMixin):
    """Extract features from text columns."""
    
    def __init__(
        self,
        text_columns: List[str] = None,
        max_features: int = 1000,
        ngram_range: Tuple[int, int] = (1, 2),
        use_tfidf: bool = True,
        extract_length_features: bool = True
    ):
        """Initialize text feature extractor.
        
        Args:
            text_columns: List of text column names.
            max_features: Maximum number of text features.
            ngram_range: Range of n-grams to extract.
            use_tfidf: Whether to use TF-IDF (vs count) vectorization.
            extract_length_features: Whether to extract text length features.
        """
        self.text_columns = text_columns or []
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.use_tfidf = use_tfidf
        self.extract_length_features = extract_length_features
        self.vectorizers_ = {}
        self.feature_names_ = []
        
    def fit(self, X: pd.DataFrame, y: Optional[np.ndarray] = None):
        """Fit the text feature extractor.
        
        Args:
            X: Feature matrix.
            y: Target vector (ignored).
            
        Returns:
            self: Fitted transformer.
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("TextFeatureExtractor requires pandas DataFrame")
        
        # Auto-detect text columns if not specified
        if not self.text_columns:
            self.text_columns = X.select_dtypes(include=['object']).columns.tolist()
        
        # Fit vectorizers for each text column
        for col in self.text_columns:
            if col in X.columns:
                text_data = X[col].fillna('').astype(str)
                
                if self.use_tfidf:
                    from sklearn.feature_extraction.text import TfidfVectorizer
                    vectorizer = TfidfVectorizer(
                        max_features=self.max_features,
                        ngram_range=self.ngram_range,
                        stop_words='english'
                    )
                else:
                    from sklearn.feature_extraction.text import CountVectorizer
                    vectorizer = CountVectorizer(
                        max_features=self.max_features,
                        ngram_range=self.ngram_range,
                        stop_words='english'
                    )
                
                vectorizer.fit(text_data)
                self.vectorizers_[col] = vectorizer
        
        return self
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform text data to numerical features.
        
        Args:
            X: Feature matrix.
            
        Returns:
            X_text: Numerical text features.
        """
        features = []
        
        # Non-text features
        non_text_cols = [col for col in X.columns if col not in self.text_columns]
        if non_text_cols:
            features.append(X[non_text_cols].select_dtypes(include=[np.number]).values)
        
        # Text features
        for col in self.text_columns:
            if col in X.columns and col in self.vectorizers_:
                text_data = X[col].fillna('').astype(str)
                
                # Vectorized features
                text_features = self.vectorizers_[col].transform(text_data).toarray()
                features.append(text_features)
                
                # Length features
                if self.extract_length_features:
                    char_count = text_data.str.len().values.reshape(-1, 1)
                    word_count = text_data.str.split().str.len().fillna(0).values.reshape(-1, 1)
                    features.extend([char_count, word_count])
        
        if features:
            return np.hstack(features)
        else:
            return np.empty((len(X), 0))


class PipelineDebugger(BaseEstimator, TransformerMixin, LoggerMixin):
    """Debug transformer that logs data shapes and statistics at each pipeline step."""
    
    def __init__(self, step_name: str = "debug", log_level: str = "INFO"):
        """Initialize pipeline debugger.
        
        Args:
            step_name: Name of the pipeline step for logging.
            log_level: Logging level.
        """
        self.step_name = step_name
        self.log_level = log_level
        
    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Optional[np.ndarray] = None):
        """Fit method (does nothing but log information).
        
        Args:
            X: Feature matrix.
            y: Target vector.
            
        Returns:
            self: Fitted transformer.
        """
        self._log_data_info(X, y, "FIT")
        return self
    
    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
        """Transform method (passes data through but logs information).
        
        Args:
            X: Feature matrix.
            
        Returns:
            X: Unchanged feature matrix.
        """
        self._log_data_info(X, None, "TRANSFORM")
        return X
    
    def _log_data_info(self, X: Union[np.ndarray, pd.DataFrame], y: Optional[np.ndarray], phase: str):
        """Log detailed information about the data."""
        info_lines = [f"=== {self.step_name} - {phase} ==="]
        
        # Shape information
        if hasattr(X, 'shape'):
            info_lines.append(f"Shape: {X.shape}")
        
        # Data type information
        if isinstance(X, pd.DataFrame):
            info_lines.append(f"Columns: {list(X.columns)}")
            info_lines.append(f"Dtypes: {X.dtypes.value_counts().to_dict()}")
            info_lines.append(f"Missing values: {X.isnull().sum().sum()}")
        else:
            info_lines.append(f"Type: {type(X).__name__}")
            if hasattr(X, 'dtype'):
                info_lines.append(f"Dtype: {X.dtype}")
        
        # Basic statistics for numerical data
        try:
            if isinstance(X, pd.DataFrame):
                numerical_cols = X.select_dtypes(include=[np.number]).columns
                if len(numerical_cols) > 0:
                    stats = X[numerical_cols].describe()
                    info_lines.append(f"Numerical stats:\n{stats}")
            elif isinstance(X, np.ndarray) and X.dtype.kind in 'biufc':
                info_lines.append(f"Mean: {np.mean(X):.4f}, Std: {np.std(X):.4f}")
                info_lines.append(f"Min: {np.min(X):.4f}, Max: {np.max(X):.4f}")
        except Exception as e:
            info_lines.append(f"Could not compute statistics: {e}")
        
        # Target information
        if y is not None:
            if hasattr(y, 'shape'):
                info_lines.append(f"Target shape: {y.shape}")
            info_lines.append(f"Target type: {type(y).__name__}")
            try:
                unique_values = len(np.unique(y))
                info_lines.append(f"Unique target values: {unique_values}")
                if unique_values <= 10:
                    value_counts = pd.Series(y).value_counts()
                    info_lines.append(f"Target distribution: {value_counts.to_dict()}")
            except:
                pass
        
        # Log all information
        for line in info_lines:
            getattr(self.logger, self.log_level.lower())(line)