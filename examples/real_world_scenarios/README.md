# Real-World ML Scenarios

End-to-end, self-contained example scripts that apply `sklearn_mastery` to
realistic business problems across six domains. Every script generates its own
synthetic data (nothing is downloaded), runs headlessly, prints a narrative
walk-through of the workflow and saves its figures under
`results/figures/examples/<script>.png`.

## Running a scenario

Run from the repository root with the project virtual environment:

```bash
# any single scenario (works from any directory; the script adds the repo root to sys.path)
MPLBACKEND=Agg .venv/bin/python examples/real_world_scenarios/business_analytics/customer_churn_prediction.py

# equivalently, as a module
MPLBACKEND=Agg .venv/bin/python -m examples.real_world_scenarios.business_analytics.customer_churn_prediction
```

`MPLBACKEND=Agg` is only needed on machines without a display; the scripts never
call `plt.show()`, so they also run fine over SSH or on CI.

To run all of them as a test suite (each script in its own subprocess, 300 s
timeout, exit code must be 0):

```bash
.venv/bin/python -m pytest tests/test_examples.py -m slow -q
```

## Runtime expectation

Every script is sized to finish in well under two minutes on a laptop (most in
3-30 s); the whole suite takes about 4-5 minutes serially. Sizes are
deliberately small (a few thousand samples, small forests, 3-fold CV, tiny
grids) so the examples stay educational rather than exhaustive. Increase the
`data_size` / `n_estimators` / `cross_validation_folds` entries in each script's
`config` dictionary to scale them up.

## Scenarios

| Domain | Script | Problem | Task type |
| --- | --- | --- | --- |
| business_analytics | `customer_churn_prediction.py` | Predict which subscribers will churn, with retention ROI | Binary classification |
| business_analytics | `fraud_detection.py` | Flag fraudulent card transactions on imbalanced data | Binary classification / anomaly detection |
| business_analytics | `market_basket_analysis.py` | Product affinity, association rules and cross-sell | Association rules / clustering |
| business_analytics | `sales_forecasting.py` | Forecast daily revenue from calendar and lag features | Regression / time series |
| finance | `algorithmic_trading.py` | Direction and return prediction for a trading strategy | Classification + regression |
| finance | `credit_scoring.py` | Loan default risk with imbalanced-learn resampling | Binary classification |
| finance | `portfolio_optimization.py` | Asset return prediction and allocation | Regression |
| finance | `risk_assessment.py` | Credit rating, probability of default, expected loss | Classification + regression |
| healthcare | `drug_discovery.py` | Molecular activity and property prediction | Classification + regression |
| healthcare | `medical_diagnosis.py` | Multi-class disease diagnosis from symptoms and labs | Multi-class classification |
| healthcare | `patient_outcome_prediction.py` | Readmission, mortality and length-of-stay | Classification + regression |
| manufacturing | `demand_forecasting.py` | Product demand forecasting for production planning | Regression / time series |
| manufacturing | `quality_control.py` | Defect detection from process sensors | Binary classification |
| manufacturing | `supply_chain_optimization.py` | Service level and inventory optimisation | Regression |
| marketing | `campaign_optimization.py` | Campaign response, ROI and spend allocation | Regression |
| marketing | `customer_segmentation.py` | Behavioural segmentation with KMeans / GMM / hierarchical | Clustering |
| marketing | `sentiment_analysis.py` | Review sentiment from bag-of-words features | Text classification |
| technology | `anomaly_detection.py` | System-monitoring anomalies and alerting | Unsupervised anomaly detection |
| technology | `natural_language_processing.py` | Text classification pipelines on synthetic corpora | Text classification |
| technology | `predictive_maintenance.py` | Equipment failure prediction from sensor telemetry | Classification + clustering |
| technology | `recommendation_systems.py` | Collaborative filtering and ranking metrics | Recommendation |

## What every scenario contains

1. Business problem definition and configuration (`config` dictionary; any key
   you pass overrides the default).
2. Synthetic data generation via `examples/real_world_scenarios/utilities/data_loaders.py`
   or an in-script generator.
3. Domain-specific feature engineering.
4. Model training and comparison using the `sklearn_mastery` model factories
   (`ClassificationModels`, `RegressionModels`, `ClusteringModels`,
   `EnsembleMethods`) and, where useful, plain scikit-learn.
5. Evaluation with `sklearn_mastery.evaluation.metrics` and the business
   helpers in `utilities/evaluation_helpers.py`.
6. Business impact / ROI analysis and a saved dashboard figure
   (`utilities/visualization_helpers.py`).

## Using the pieces from your own code

```python
from examples.real_world_scenarios.utilities.data_loaders import DataLoader
from examples.real_world_scenarios.business_analytics.customer_churn_prediction import CustomerChurnPredictor

loader = DataLoader(random_state=0)
X, y = loader.load_customer_churn_data(n_samples=2000)

predictor = CustomerChurnPredictor(config={"data_size": 2000, "hyperparameter_tuning": False})
results = predictor.run_complete_analysis()
```

Import the package from the repository root (or with the repository root on
`PYTHONPATH`); `examples` is a plain namespace package.

## Shared utilities

| Module | Contents |
| --- | --- |
| `utilities/data_loaders.py` | `DataLoader`: synthetic churn, fraud, sales, segmentation, recommendation and diagnosis datasets; `train_test_split` that stratifies automatically for categorical targets |
| `utilities/evaluation_helpers.py` | `BusinessMetricsCalculator`, `ModelPerformanceEvaluator`, `BusinessReportGenerator`, lift/gain and prediction-interval helpers |
| `utilities/visualization_helpers.py` | `BusinessVisualizer` dashboards (matplotlib, optional Plotly) and `ModelVisualizer` |

## Adding a scenario

1. Copy an existing script in the closest domain and keep its structure
   (problem statement docstring, `config` defaults, `run_complete_analysis()`, `main()`).
2. Generate data in-script or extend `DataLoader`; never download.
3. Keep the default configuration small enough to finish in under two minutes.
4. Save figures instead of showing them, and guard execution with
   `if __name__ == "__main__":`.
5. Run `pytest tests/test_examples.py -m slow` before opening a PR; the test
   discovers new scripts automatically.
