"""Custom transformers wrapper module."""

import sys
from pathlib import Path

# Handle imports that work in both package and direct import contexts
try:
    from ..pipelines.custom_transformers import *
except ImportError:
    # Fallback for direct imports outside package context
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from pipelines.custom_transformers import *

__all__ = [
    "OutlierRemover",
    "FeatureInteractionCreator",
    "DomainSpecificEncoder",
    "AdvancedImputer",
    "FeatureScaler",
    "TimeSeriesFeatureCreator",
    "TextFeatureExtractor",
    "PipelineDebugger"
]
