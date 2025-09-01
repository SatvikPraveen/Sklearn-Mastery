.
├── .DS_Store
├── .github
│   └── workflows
│       ├── ci.yml
│       └── results_validation.yml
├── .gitignore
├── .pre-commit-config.yaml
├── config
│   ├── __init__.py
│   ├── logging_config.py
│   ├── results_config.yaml
│   └── settings.py
├── Dockerfile
├── docs
│   ├── .DS_Store
│   ├── algorithm_guides
│   │   ├── classification.md
│   │   ├── clustering.md
│   │   ├── dimensionality_reduction.md
│   │   ├── ensemble_methods.md
│   │   └── regression.md
│   ├── api_reference
│   │   ├── data.md
│   │   ├── evaluation.md
│   │   ├── index.md
│   │   ├── models.md
│   │   ├── pipelines.md
│   │   └── utils.md
│   ├── CONTRIBUTING.md
│   ├── deployment.md
│   ├── examples
│   │   └── README.md
│   ├── FAQ.md
│   ├── index.md
│   ├── troubleshooting.md
│   └── tutorials
│       ├── data_preprocessing.md
│       ├── getting_started.md
│       └── model_selection.md
├── examples
│   └── real_world_scenarios
│       ├── __init__.py
│       ├── business_analytics
│       │   ├── custome_churn_prediction.py
│       │   ├── fraud_detection.py
│       │   ├── market_basket_analysis.py
│       │   └── sales_forecasting.py
│       ├── finance
│       │   ├── algorithmic_trading.py
│       │   ├── credit_scoring.py
│       │   ├── portfolio_optimization.py
│       │   └── risk_assessment.py
│       ├── healthcare
│       │   ├── drug_discovery.py
│       │   ├── medical_diagnosis.py
│       │   └── patient_outcome_prediction.py
│       ├── manufacturing
│       │   ├── demand_forecasting.py
│       │   ├── quality_control.py
│       │   └── supply_chain_optimization.py
│       ├── marketing
│       │   ├── campaign_optimization.py
│       │   ├── customer_segmentation.py
│       │   └── sentiment_analysis.py
│       ├── README.md
│       ├── technology
│       │   ├── anomaly_detection.py
│       │   ├── natural_language_processing.py
│       │   ├── predictive_maintenance.py
│       │   └── recommendation_systems.py
│       └── utilities
│           ├── data_loaders.py
│           ├── evaluation_helpers.py
│           └── visualization_helpers.py
├── Makefile
├── mkdocs.yml
├── notebooks
│   ├── .DS_Store
│   ├── 01_data_generation_showcase.ipynb
│   ├── 02_preprocessing_pipelines.ipynb
│   ├── 03_supervised_learning.ipynb
│   ├── 04_unsupervised_learning.ipynb
│   ├── 05_ensemble_methods.ipynb
│   ├── 06_model_selection_tuning.ipynb
│   └── 07_advanced_techniques.ipynb
├── PROJECT_STRUCTURE.md
├── README.md
├── requirements-docs.txt
├── requirements-minimal.txt
├── requirements.txt
├── results
│   ├── figures
│   ├── models
│   └── reports
├── setup.py
├── src
│   ├── __init__.py
│   ├── .DS_Store
│   ├── cli.py
│   ├── data
│   │   ├── __init__.py
│   │   ├── generators.py
│   │   ├── preprocessors.py
│   │   └── validators.py
│   ├── evaluation
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   ├── statistical_tests.py
│   │   ├── utils.py
│   │   └── visualization.py
│   ├── models
│   │   ├── __init__.py
│   │   ├── .DS_Store
│   │   ├── ensemble
│   │   │   ├── __init__.py
│   │   │   └── methods.py
│   │   ├── supervised
│   │   │   ├── __init__.py
│   │   │   ├── classification.py
│   │   │   └── regression.py
│   │   └── unsupervised
│   │       ├── __init__.py
│   │       ├── clustering.py
│   │       └── dimensionality_reduction.py
│   ├── pipelines
│   │   ├── __init__.py
│   │   ├── custom_transformers.py
│   │   ├── model_selection.py
│   │   └── pipeline_factory.py
│   └── utils
│       ├── __init__.py
│       ├── decorators.py
│       └── helpers.py
└── tests
    ├── .DS_Store
    ├── test_data
    │   ├── __init__.py
    │   ├── test_generators.py
    │   ├── test_preprocessors.py
    │   └── test_validators.py
    ├── test_models
    │   ├── __init__.py
    │   ├── test_classification.py
    │   ├── test_clustering.py
    │   ├── test_dimensionality_reduction.py
    │   ├── test_ensemble.py
    │   └── test_regression.py
    ├── test_pipelines
    │   ├── __init__.py
    │   ├── test_custom_transformers.py
    │   ├── test_feature_union.py
    │   ├── test_model_selection.py
    │   └── test_pipeline_factory.py
    └── test_utils
        ├── __init__.py
        ├── test_evaluation.py
        └── test_visualization.py

37 directories, 121 files
