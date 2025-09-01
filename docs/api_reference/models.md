# Models API Reference

Complete API documentation for all machine learning models in the project.

## Base Classes

### ClassificationModel

Base class for all classification models.

```python
class ClassificationModel:
    """Base class for classification models."""

    def train(self, X, y):
        """Train the classification model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training feature matrix.
        y : array-like of shape (n_samples,)
            Training target labels.
        """

    def predict(self, X):
        """Make class predictions.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix for prediction.

        Returns
        -------
        array of shape (n_samples,)
            Predicted class labels.
        """

    def predict_proba(self, X):
        """Predict class probabilities.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix for prediction.

        Returns
        -------
        array of shape (n_samples, n_classes)
            Class probabilities.
        """

    def evaluate(self, X, y):
        """Evaluate model performance.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.
        y : array-like of shape (n_samples,)
            True labels.

        Returns
        -------
        dict
            Dictionary containing evaluation metrics:
            - 'accuracy': Overall accuracy
            - 'precision': Precision score
            - 'recall': Recall score
            - 'f1': F1 score
            - 'confusion_matrix': Confusion matrix
            - 'classification_report': Detailed classification report
        """

    def cross_validate(self, X, y, cv=5):
        """Perform cross-validation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.
        y : array-like of shape (n_samples,)
            Target labels.
        cv : int, default=5
            Number of cross-validation folds.

        Returns
        -------
        dict
            Cross-validation results containing:
            - 'test_score': Test scores for each fold
            - 'train_score': Training scores for each fold
            - 'fit_time': Fitting time for each fold
        """

    def save_model(self, filepath):
        """Save trained model to file.

        Parameters
        ----------
        filepath : str
            Path to save the model.
        """

    def load_model(self, filepath):
        """Load trained model from file.

        Parameters
        ----------
        filepath : str
            Path to load the model from.
        """
```

### RegressionModel

Base class for all regression models.

```python
class RegressionModel:
    """Base class for regression models."""

    def train(self, X, y):
        """Train the regression model."""

    def predict(self, X):
        """Make predictions."""

    def evaluate(self, X, y):
        """Evaluate model performance.

        Returns
        -------
        dict
            Dictionary containing evaluation metrics:
            - 'mse': Mean Squared Error
            - 'rmse': Root Mean Squared Error
            - 'mae': Mean Absolute Error
            - 'r2': R-squared score
        """
```

## Classification Models

### LogisticRegressionModel

```python
class LogisticRegressionModel(ClassificationModel):
    """Logistic Regression classifier with enhanced functionality.

    Parameters
    ----------
    penalty : {'l1', 'l2', 'elasticnet', 'none'}, default='l2'
        Norm of the penalty.
    dual : bool, default=False
        Dual or primal formulation.
    tol : float, default=1e-4
        Tolerance for stopping criteria.
    C : float, default=1.0
        Inverse of regularization strength.
    fit_intercept : bool, default=True
        Whether to fit intercept.
    intercept_scaling : float, default=1
        Intercept scaling factor.
    class_weight : dict or 'balanced', default=None
        Weights associated with classes.
    random_state : int, default=None
        Random state for reproducibility.
    solver : {'lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'}, default='lbfgs'
        Algorithm to use in optimization.
    max_iter : int, default=100
        Maximum number of iterations.
    multi_class : {'auto', 'ovr', 'multinomial'}, default='auto'
        Multi-class strategy.
    verbose : int, default=0
        Verbosity level.
    warm_start : bool, default=False
        Whether to reuse solution of previous call.
    n_jobs : int, default=None
        Number of CPU cores to use.
    l1_ratio : float, default=None
        Elastic-Net mixing parameter.

    Examples
    --------
    >>> from src.models.supervised.classification import LogisticRegressionModel
    >>> model = LogisticRegressionModel(C=1.0, random_state=42)
    >>> model.train(X_train, y_train)
    >>> predictions = model.predict(X_test)
    >>> probabilities = model.predict_proba(X_test)
    """

    def get_coefficients(self):
        """Get model coefficients.

        Returns
        -------
        ndarray of shape (n_features,) or (n_classes, n_features)
            Model coefficients.
        """

    def get_intercept(self):
        """Get model intercept.

        Returns
        -------
        ndarray of shape (1,) or (n_classes,)
            Model intercept.
        """

    def tune_hyperparameters(self, X, y, param_grid, cv=5):
        """Tune hyperparameters using grid search.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.
        y : array-like of shape (n_samples,)
            Target labels.
        param_grid : dict
            Parameter grid for search.
        cv : int, default=5
            Number of cross-validation folds.

        Returns
        -------
        tuple
            (best_params, best_score)
        """
```

### RandomForestClassifierModel

```python
class RandomForestClassifierModel(ClassificationModel):
    """Random Forest classifier with enhanced functionality.

    Parameters
    ----------
    n_estimators : int, default=100
        Number of trees in the forest.
    criterion : {'gini', 'entropy', 'log_loss'}, default='gini'
        Function to measure quality of split.
    max_depth : int, default=None
        Maximum depth of trees.
    min_samples_split : int or float, default=2
        Minimum number of samples required to split.
    min_samples_leaf : int or float, default=1
        Minimum number of samples required in leaf.
    min_weight_fraction_leaf : float, default=0.0
        Minimum weighted fraction in leaf.
    max_features : {'sqrt', 'log2', None} or int or float, default='sqrt'
        Number of features for best split.
    max_leaf_nodes : int, default=None
        Maximum number of leaf nodes.
    min_impurity_decrease : float, default=0.0
        Minimum impurity decrease for split.
    bootstrap : bool, default=True
        Whether bootstrap samples are used.
    oob_score : bool, default=False
        Whether to use out-of-bag samples for score.
    n_jobs : int, default=None
        Number of jobs for parallel computation.
    random_state : int, default=None
        Random state for reproducibility.
    verbose : int, default=0
        Verbosity level.
    warm_start : bool, default=False
        Whether to reuse previous solution.
    class_weight : {'balanced', 'balanced_subsample'} or dict, default=None
        Weights associated with classes.
    ccp_alpha : float, default=0.0
        Complexity parameter for pruning.
    max_samples : int or float, default=None
        Number of samples to draw for training.

    Examples
    --------
    >>> model = RandomForestClassifierModel(n_estimators=100, random_state=42)
    >>> model.train(X_train, y_train)
    >>> importance = model.get_feature_importance()
    >>> oob_score = model.get_oob_score()
    """

    def get_feature_importance(self):
        """Get feature importance scores.

        Returns
        -------
        ndarray of shape (n_features,)
            Feature importance scores (normalized to sum to 1).
        """

    def get_oob_score(self):
        """Get out-of-bag score.

        Returns
        -------
        float
            Out-of-bag score (only if oob_score=True).
        """

    def get_top_features(self, k=10):
        """Get top-k most important features.

        Parameters
        ----------
        k : int, default=10
            Number of top features to return.

        Returns
        -------
        list of tuples
            List of (feature_index, importance_score) tuples.
        """
```

### SVMClassifierModel

```python
class SVMClassifierModel(ClassificationModel):
    """Support Vector Machine classifier.

    Parameters
    ----------
    C : float, default=1.0
        Regularization parameter.
    kernel : {'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'}, default='rbf'
        Kernel type.
    degree : int, default=3
        Degree for polynomial kernel.
    gamma : {'scale', 'auto'} or float, default='scale'
        Kernel coefficient.
    coef0 : float, default=0.0
        Independent term in kernel function.
    shrinking : bool, default=True
        Whether to use shrinking heuristic.
    probability : bool, default=False
        Whether to enable probability estimates.
    tol : float, default=1e-3
        Tolerance for stopping criterion.
    cache_size : float, default=200
        Size of kernel cache (in MB).
    class_weight : dict or 'balanced', default=None
        Weights associated with classes.
    verbose : bool, default=False
        Enable verbose output.
    max_iter : int, default=-1
        Hard limit on iterations.
    decision_function_shape : {'ovo', 'ovr'}, default='ovr'
        Decision function shape.
    break_ties : bool, default=False
        Break ties according to confidence values.
    random_state : int, default=None
        Random state for reproducibility.

    Examples
    --------
    >>> model = SVMClassifierModel(kernel='rbf', C=1.0, random_state=42)
    >>> model.train(X_train, y_train)
    >>> decision_scores = model.decision_function(X_test)
    """

    def decision_function(self, X):
        """Get decision function values.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.

        Returns
        -------
        ndarray
            Decision function values.
        """

    def get_support_vectors(self):
        """Get support vectors.

        Returns
        -------
        ndarray of shape (n_support_vectors, n_features)
            Support vectors.
        """
```

## Regression Models

### LinearRegressionModel

```python
class LinearRegressionModel(RegressionModel):
    """Linear Regression with enhanced functionality.

    Parameters
    ----------
    fit_intercept : bool, default=True
        Whether to fit intercept.
    copy_X : bool, default=True
        Whether to copy X.
    n_jobs : int, default=None
        Number of jobs for computation.
    positive : bool, default=False
        Whether to force positive coefficients.

    Examples
    --------
    >>> model = LinearRegressionModel()
    >>> model.train(X_train, y_train)
    >>> coefficients = model.get_coefficients()
    >>> intercept = model.get_intercept()
    """

    def get_coefficients(self):
        """Get regression coefficients."""

    def get_intercept(self):
        """Get regression intercept."""
```

### RandomForestRegressorModel

```python
class RandomForestRegressorModel(RegressionModel):
    """Random Forest regressor with enhanced functionality.

    Similar parameters to RandomForestClassifierModel but for regression.

    Examples
    --------
    >>> model = RandomForestRegressorModel(n_estimators=100, random_state=42)
    >>> model.train(X_train, y_train)
    >>> importance = model.get_feature_importance()
    """

    def get_feature_importance(self):
        """Get feature importance scores."""
```

## Clustering Models

### KMeansModel

```python
class KMeansModel:
    """K-Means clustering with enhanced functionality.

    Parameters
    ----------
    n_clusters : int, default=8
        Number of clusters.
    init : {'k-means++', 'random'} or ndarray, default='k-means++'
        Initialization method.
    n_init : int, default=10
        Number of random initializations.
    max_iter : int, default=300
        Maximum number of iterations.
    tol : float, default=1e-4
        Tolerance for convergence.
    verbose : int, default=0
        Verbosity mode.
    random_state : int, default=None
        Random state for reproducibility.
    copy_x : bool, default=True
        Whether to copy input data.
    algorithm : {'lloyd', 'elkan', 'auto', 'full'}, default='lloyd'
        K-means algorithm to use.

    Examples
    --------
    >>> model = KMeansModel(n_clusters=3, random_state=42)
    >>> model.fit(X)
    >>> labels = model.predict(X)
    >>> centers = model.get_cluster_centers()
    >>> inertia = model.get_inertia()
    """

    def fit(self, X):
        """Fit the clustering model."""

    def predict(self, X):
        """Predict cluster labels."""

    def fit_predict(self, X):
        """Fit model and predict labels."""

    def get_cluster_centers(self):
        """Get cluster centers."""

    def get_inertia(self):
        """Get within-cluster sum of squared distances."""

    def transform(self, X):
        """Transform X to cluster-distance space."""
```

## Dimensionality Reduction Models

### PCAModel

```python
class PCAModel:
    """Principal Component Analysis with enhanced functionality.

    Parameters
    ----------
    n_components : int, float or 'mle', default=None
        Number of components to keep.
    copy : bool, default=True
        Whether to copy input data.
    whiten : bool, default=False
        Whether to whiten components.
    svd_solver : {'auto', 'full', 'arpack', 'randomized'}, default='auto'
        SVD solver to use.
    tol : float, default=0.0
        Tolerance for singular values.
    iterated_power : int or 'auto', default='auto'
        Number of iterations for randomized SVD.
    n_oversamples : int, default=10
        Additional number of random vectors for randomized SVD.
    power_iteration_normalizer : {'auto', 'QR', 'LU', 'none'}, default='auto'
        Power iteration normalizer.
    random_state : int, default=None
        Random state for reproducibility.

    Examples
    --------
    >>> model = PCAModel(n_components=2, random_state=42)
    >>> X_transformed = model.fit_transform(X)
    >>> components = model.get_components()
    >>> explained_variance = model.get_explained_variance_ratio()
    """

    def fit(self, X):
        """Fit PCA model."""

    def transform(self, X):
        """Transform data to lower dimension."""

    def fit_transform(self, X):
        """Fit model and transform data."""

    def inverse_transform(self, X):
        """Transform data back to original space."""

    def get_components(self):
        """Get principal components."""

    def get_explained_variance(self):
        """Get explained variance."""

    def get_explained_variance_ratio(self):
        """Get explained variance ratio."""
```

## Ensemble Models

### VotingEnsemble

```python
class VotingEnsemble:
    """Voting ensemble classifier/regressor.

    Parameters
    ----------
    estimators : list of (str, estimator) tuples
        List of (name, estimator) tuples.
    voting : {'hard', 'soft'}, default='hard'
        Voting strategy.
    weights : array-like, default=None
        Weights for estimators.
    n_jobs : int, default=None
        Number of jobs for parallel computation.
    flatten_transform : bool, default=True
        Whether to flatten transform output.
    verbose : bool, default=False
        Enable verbose output.

    Examples
    --------
    >>> estimators = [('rf', RandomForestClassifier()), ('svm', SVC())]
    >>> ensemble = VotingEnsemble(estimators=estimators, voting='soft')
    >>> ensemble.fit(X_train, y_train)
    >>> predictions = ensemble.predict(X_test)
    """

    def fit(self, X, y):
        """Fit ensemble."""

    def predict(self, X):
        """Make ensemble predictions."""

    def predict_proba(self, X):
        """Get ensemble prediction probabilities."""

    def get_individual_predictions(self, X):
        """Get predictions from individual estimators."""
```

## Model Selection Utilities

### ModelEvaluator

```python
class ModelEvaluator:
    """Comprehensive model evaluation utility.

    Parameters
    ----------
    task_type : {'classification', 'regression'}
        Type of machine learning task.
    custom_metrics : dict, default=None
        Custom evaluation metrics.

    Examples
    --------
    >>> evaluator = ModelEvaluator(task_type='classification')
    >>> metrics = evaluator.evaluate(model, X_test, y_test)
    >>> report = evaluator.detailed_report(model, X_test, y_test)
    """

    def evaluate(self, model, X, y):
        """Evaluate model performance."""

    def detailed_report(self, model, X, y):
        """Generate detailed evaluation report."""

    def compare_models(self, models_dict, X, y):
        """Compare multiple models."""
```

## Common Parameters

### Shared Parameters Across Models

- **random_state** : int, default=None

  - Controls randomness for reproducible results
  - Set to integer for deterministic behavior

- **n_jobs** : int, default=None

  - Number of CPU cores to use for parallel computation
  - -1 uses all available cores

- **verbose** : int or bool, default=0 or False
  - Controls verbosity of output during training

### Model-Specific Parameter Patterns

- **n_estimators** : Used in ensemble methods (Random Forest, Gradient Boosting)
- **max_iter** : Maximum iterations for iterative algorithms
- **tol** : Tolerance for convergence criteria
- **C** : Regularization parameter (inverse of regularization strength)
- **alpha** : Regularization parameter (direct regularization strength)
