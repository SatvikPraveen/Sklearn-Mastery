# Deployment Guide

**File Location:** `docs/deployment.md`

This guide covers various deployment strategies for MyML models, from simple local deployments to scalable cloud solutions.

## Table of Contents

- [Local Deployment](#local-deployment)
- [Web API Deployment](#web-api-deployment)
- [Cloud Deployment](#cloud-deployment)
- [Container Deployment](#container-deployment)
- [Model Serving](#model-serving)
- [Monitoring and Maintenance](#monitoring-and-maintenance)
- [Best Practices](#best-practices)

## Local Deployment

### Simple Model Serialization

The most basic deployment involves saving and loading trained models:

```python
from myml.utils.io_utils import save_model, load_model
from myml.algorithms import RandomForestClassifier
import joblib

# Train and save model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Save model
save_model(rf, 'models/production_model.pkl')

# Load model in production
model = load_model('models/production_model.pkl')
predictions = model.predict(X_new)
```

### Batch Prediction Script

Create a script for batch processing:

```python
#!/usr/bin/env python3
"""
Batch prediction script for MyML models.
Usage: python batch_predict.py --model model.pkl --input data.csv --output predictions.csv
"""

import argparse
import pandas as pd
from myml.utils.io_utils import load_model

def main():
    parser = argparse.ArgumentParser(description='Batch prediction with MyML')
    parser.add_argument('--model', required=True, help='Path to saved model')
    parser.add_argument('--input', required=True, help='Input data CSV')
    parser.add_argument('--output', required=True, help='Output predictions CSV')
    parser.add_argument('--batch-size', type=int, default=1000,
                       help='Batch size for processing')

    args = parser.parse_args()

    # Load model and data
    model = load_model(args.model)
    data = pd.read_csv(args.input)

    # Process in batches
    predictions = []
    for i in range(0, len(data), args.batch_size):
        batch = data.iloc[i:i+args.batch_size]
        batch_pred = model.predict(batch)
        predictions.extend(batch_pred)

    # Save predictions
    result_df = data.copy()
    result_df['prediction'] = predictions
    result_df.to_csv(args.output, index=False)

    print(f"Processed {len(data)} samples, saved to {args.output}")

if __name__ == '__main__':
    main()
```

## Web API Deployment

### Flask API

Simple REST API using Flask:

```python
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from myml.utils.io_utils import load_model

app = Flask(__name__)

# Load model at startup
MODEL = load_model('models/production_model.pkl')

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({'status': 'healthy', 'model_loaded': MODEL is not None})

@app.route('/predict', methods=['POST'])
def predict():
    """
    Prediction endpoint.

    Expected JSON format:
    {
        "instances": [
            {"feature1": 1.0, "feature2": 2.0, ...},
            {"feature1": 1.5, "feature2": 2.5, ...}
        ]
    }
    """
    try:
        data = request.get_json()

        if 'instances' not in data:
            return jsonify({'error': 'Missing instances field'}), 400

        # Convert to DataFrame
        df = pd.DataFrame(data['instances'])

        # Make predictions
        predictions = MODEL.predict(df)
        probabilities = None

        # Get probabilities if available
        if hasattr(MODEL, 'predict_proba'):
            probabilities = MODEL.predict_proba(df).tolist()

        response = {
            'predictions': predictions.tolist(),
            'probabilities': probabilities
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """Batch prediction endpoint for large datasets."""
    try:
        data = request.get_json()
        instances = data['instances']
        batch_size = data.get('batch_size', 1000)

        all_predictions = []
        all_probabilities = []

        # Process in batches
        for i in range(0, len(instances), batch_size):
            batch = instances[i:i+batch_size]
            df_batch = pd.DataFrame(batch)

            batch_pred = MODEL.predict(df_batch)
            all_predictions.extend(batch_pred.tolist())

            if hasattr(MODEL, 'predict_proba'):
                batch_prob = MODEL.predict_proba(df_batch)
                all_probabilities.extend(batch_prob.tolist())

        response = {
            'predictions': all_predictions,
            'probabilities': all_probabilities if all_probabilities else None
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
```

### FastAPI Implementation

More modern API using FastAPI:

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import pandas as pd
from myml.utils.io_utils import load_model

app = FastAPI(title="MyML Model API", version="1.0.0")

# Load model at startup
MODEL = load_model('models/production_model.pkl')

class PredictionRequest(BaseModel):
    instances: List[Dict[str, Any]]

class BatchPredictionRequest(BaseModel):
    instances: List[Dict[str, Any]]
    batch_size: Optional[int] = 1000

class PredictionResponse(BaseModel):
    predictions: List[Any]
    probabilities: Optional[List[List[float]]] = None

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "model_loaded": MODEL is not None}

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Single prediction endpoint."""
    try:
        df = pd.DataFrame(request.instances)
        predictions = MODEL.predict(df)

        probabilities = None
        if hasattr(MODEL, 'predict_proba'):
            probabilities = MODEL.predict_proba(df).tolist()

        return PredictionResponse(
            predictions=predictions.tolist(),
            probabilities=probabilities
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", response_model=PredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """Batch prediction endpoint."""
    try:
        all_predictions = []
        all_probabilities = []

        for i in range(0, len(request.instances), request.batch_size):
            batch = request.instances[i:i+request.batch_size]
            df_batch = pd.DataFrame(batch)

            batch_pred = MODEL.predict(df_batch)
            all_predictions.extend(batch_pred.tolist())

            if hasattr(MODEL, 'predict_proba'):
                batch_prob = MODEL.predict_proba(df_batch)
                all_probabilities.extend(batch_prob.tolist())

        return PredictionResponse(
            predictions=all_predictions,
            probabilities=all_probabilities if all_probabilities else None
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run with: uvicorn main:app --host 0.0.0.0 --port 8000
```

## Container Deployment

### Dockerfile

```dockerfile
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash app
RUN chown -R app:app /app
USER app

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose

```yaml
version: "3.8"

services:
  myml-api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
      - ./logs:/app/logs
    environment:
      - MODEL_PATH=/app/models/production_model.pkl
      - LOG_LEVEL=INFO
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 30s

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - myml-api
    restart: unless-stopped

  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
    restart: unless-stopped
```

## Cloud Deployment

### AWS Deployment

#### Using AWS Lambda

```python
import json
import boto3
from myml.utils.io_utils import load_model

# Lambda function
def lambda_handler(event, context):
    """AWS Lambda function for model inference."""

    # Load model (consider using S3 for large models)
    model = load_model('model.pkl')

    # Parse input
    body = json.loads(event['body'])
    instances = body['instances']

    # Make predictions
    df = pd.DataFrame(instances)
    predictions = model.predict(df)

    return {
        'statusCode': 200,
        'body': json.dumps({
            'predictions': predictions.tolist()
        }),
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*'
        }
    }
```

#### AWS SageMaker Deployment

```python
import sagemaker
from sagemaker.sklearn.model import SKLearnModel

# Create SageMaker model
sklearn_model = SKLearnModel(
    model_data='s3://your-bucket/model.tar.gz',
    role='arn:aws:iam::your-account:role/SageMakerRole',
    entry_point='inference.py',
    framework_version='1.0-1',
    py_version='py3'
)

# Deploy to endpoint
predictor = sklearn_model.deploy(
    initial_instance_count=1,
    instance_type='ml.t2.medium',
    endpoint_name='myml-model-endpoint'
)

# Make predictions
result = predictor.predict(test_data)
```

### Google Cloud Platform

#### Cloud Run Deployment

```python
# app.py for Cloud Run
import os
from flask import Flask, request, jsonify
from myml.utils.io_utils import load_model

app = Flask(__name__)

# Load model from Cloud Storage
MODEL_PATH = os.environ.get('MODEL_PATH', 'models/model.pkl')
model = load_model(MODEL_PATH)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    predictions = model.predict(pd.DataFrame(data['instances']))
    return jsonify({'predictions': predictions.tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
```

#### Kubernetes Deployment

```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: myml-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: myml-api
  template:
    metadata:
      labels:
        app: myml-api
    spec:
      containers:
        - name: myml-api
          image: gcr.io/your-project/myml-api:latest
          ports:
            - containerPort: 8000
          env:
            - name: MODEL_PATH
              value: "/models/model.pkl"
          resources:
            requests:
              memory: "512Mi"
              cpu: "250m"
            limits:
              memory: "1Gi"
              cpu: "500m"
---
apiVersion: v1
kind: Service
metadata:
  name: myml-service
spec:
  selector:
    app: myml-api
  ports:
    - port: 80
      targetPort: 8000
  type: LoadBalancer
```

## Model Serving

### TensorFlow Serving Alternative

```python
# Custom model server
import grpc
from concurrent import futures
import time
import model_pb2_grpc
import model_pb2
from myml.utils.io_utils import load_model

class ModelServer(model_pb2_grpc.ModelServiceServicer):
    def __init__(self):
        self.model = load_model('models/production_model.pkl')

    def Predict(self, request, context):
        # Convert gRPC request to pandas DataFrame
        data = []
        for instance in request.instances:
            data.append({
                'feature1': instance.feature1,
                'feature2': instance.feature2,
                # Add more features as needed
            })

        df = pd.DataFrame(data)
        predictions = self.model.predict(df)

        response = model_pb2.PredictResponse()
        for pred in predictions:
            response.predictions.append(pred)

        return response

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    model_pb2_grpc.add_ModelServiceServicer_to_server(ModelServer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    print("Model server started on port 50051")
    server.wait_for_termination()

if __name__ == '__main__':
    serve()
```

### Model Versioning System

```python
# model_registry.py
import os
import json
from datetime import datetime
from myml.utils.io_utils import load_model, save_model

class ModelRegistry:
    def __init__(self, registry_path='model_registry'):
        self.registry_path = registry_path
        os.makedirs(registry_path, exist_ok=True)
        self.metadata_file = os.path.join(registry_path, 'metadata.json')
        self._load_metadata()

    def _load_metadata(self):
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {'models': {}, 'active_model': None}

    def register_model(self, model, version, metrics=None):
        """Register a new model version."""
        model_path = os.path.join(self.registry_path, f'model_v{version}.pkl')
        save_model(model, model_path)

        self.metadata['models'][version] = {
            'path': model_path,
            'created_at': datetime.now().isoformat(),
            'metrics': metrics or {}
        }

        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)

        print(f"Model version {version} registered successfully")

    def load_model_version(self, version):
        """Load specific model version."""
        if version not in self.metadata['models']:
            raise ValueError(f"Model version {version} not found")

        model_path = self.metadata['models'][version]['path']
        return load_model(model_path)

    def set_active_model(self, version):
        """Set active model version."""
        if version not in self.metadata['models']:
            raise ValueError(f"Model version {version} not found")

        self.metadata['active_model'] = version
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)

    def get_active_model(self):
        """Get currently active model."""
        if not self.metadata['active_model']:
            raise ValueError("No active model set")

        return self.load_model_version(self.metadata['active_model'])

# Usage example
registry = ModelRegistry()
registry.register_model(trained_model, 'v1.0', metrics={'accuracy': 0.95})
registry.set_active_model('v1.0')
active_model = registry.get_active_model()
```

## Monitoring and Maintenance

### Performance Monitoring

```python
# monitoring.py
import time
import psutil
import logging
from functools import wraps
from prometheus_client import Counter, Histogram, generate_latest

# Metrics
REQUEST_COUNT = Counter('model_requests_total', 'Total requests')
REQUEST_LATENCY = Histogram('model_request_duration_seconds', 'Request latency')
PREDICTION_COUNT = Counter('model_predictions_total', 'Total predictions')

def monitor_performance(func):
    """Decorator to monitor model performance."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()

        try:
            result = func(*args, **kwargs)
            REQUEST_COUNT.inc()
            PREDICTION_COUNT.inc(len(result) if hasattr(result, '__len__') else 1)
            return result
        except Exception as e:
            logging.error(f"Prediction error: {e}")
            raise
        finally:
            REQUEST_LATENCY.observe(time.time() - start_time)

    return wrapper

class SystemMonitor:
    def __init__(self):
        self.start_time = time.time()

    def get_system_stats(self):
        """Get system resource usage."""
        return {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_percent': psutil.disk_usage('/').percent,
            'uptime': time.time() - self.start_time
        }

    def health_check(self, model):
        """Perform health check on model."""
        try:
            # Simple prediction test
            test_data = [[0] * 4]  # Adjust based on your model
            _ = model.predict(test_data)
            return {'status': 'healthy', 'timestamp': time.time()}
        except Exception as e:
            return {'status': 'unhealthy', 'error': str(e), 'timestamp': time.time()}

monitor = SystemMonitor()
```

### A/B Testing Framework

```python
# ab_testing.py
import random
import json
from datetime import datetime

class ABTestManager:
    def __init__(self):
        self.experiments = {}

    def create_experiment(self, name, model_a, model_b, traffic_split=0.5):
        """Create A/B test experiment."""
        self.experiments[name] = {
            'model_a': model_a,
            'model_b': model_b,
            'traffic_split': traffic_split,
            'results': {'a': [], 'b': []},
            'created_at': datetime.now().isoformat()
        }

    def get_model_for_request(self, experiment_name, user_id=None):
        """Route request to appropriate model."""
        if experiment_name not in self.experiments:
            raise ValueError(f"Experiment {experiment_name} not found")

        exp = self.experiments[experiment_name]

        # Use user_id for consistent routing, otherwise random
        if user_id:
            hash_val = hash(user_id) % 100
            use_model_a = hash_val < (exp['traffic_split'] * 100)
        else:
            use_model_a = random.random() < exp['traffic_split']

        model_variant = 'a' if use_model_a else 'b'
        model = exp['model_a'] if use_model_a else exp['model_b']

        return model, model_variant

    def record_result(self, experiment_name, variant, prediction, actual=None):
        """Record experiment result."""
        result = {
            'timestamp': datetime.now().isoformat(),
            'prediction': prediction,
            'actual': actual
        }
        self.experiments[experiment_name]['results'][variant].append(result)

    def get_experiment_stats(self, experiment_name):
        """Get experiment statistics."""
        if experiment_name not in self.experiments:
            return None

        exp = self.experiments[experiment_name]
        stats = {}

        for variant in ['a', 'b']:
            results = exp['results'][variant]
            stats[variant] = {
                'total_predictions': len(results),
                'avg_prediction': sum(r['prediction'] for r in results) / len(results) if results else 0
            }

            if any(r['actual'] is not None for r in results):
                actual_results = [r for r in results if r['actual'] is not None]
                if actual_results:
                    accuracy = sum(1 for r in actual_results if r['prediction'] == r['actual']) / len(actual_results)
                    stats[variant]['accuracy'] = accuracy

        return stats

# Usage
ab_manager = ABTestManager()
ab_manager.create_experiment('model_comparison', model_v1, model_v2, 0.3)
```

## Best Practices

### Production Checklist

```python
# production_checklist.py
import os
import logging
from myml.utils.io_utils import load_model

class ProductionValidator:
    """Validate model readiness for production."""

    def __init__(self, model_path):
        self.model_path = model_path
        self.checks = []

    def validate_model_loading(self):
        """Check if model loads correctly."""
        try:
            model = load_model(self.model_path)
            self.checks.append(("Model Loading", True, "Model loaded successfully"))
            return model
        except Exception as e:
            self.checks.append(("Model Loading", False, f"Failed to load model: {e}"))
            return None

    def validate_prediction_format(self, model, sample_data):
        """Validate prediction output format."""
        try:
            predictions = model.predict(sample_data)
            if hasattr(predictions, '__len__') and len(predictions) == len(sample_data):
                self.checks.append(("Prediction Format", True, "Predictions have correct shape"))
                return True
            else:
                self.checks.append(("Prediction Format", False, "Prediction shape mismatch"))
                return False
        except Exception as e:
            self.checks.append(("Prediction Format", False, f"Prediction failed: {e}"))
            return False

    def validate_environment(self):
        """Check environment configuration."""
        required_vars = ['MODEL_PATH', 'LOG_LEVEL']
        missing_vars = [var for var in required_vars if not os.environ.get(var)]

        if not missing_vars:
            self.checks.append(("Environment", True, "All required environment variables set"))
            return True
        else:
            self.checks.append(("Environment", False, f"Missing variables: {missing_vars}"))
            return False

    def validate_logging(self):
        """Check logging configuration."""
        try:
            logging.info("Test log message")
            self.checks.append(("Logging", True, "Logging configured correctly"))
            return True
        except Exception as e:
            self.checks.append(("Logging", False, f"Logging error: {e}"))
            return False

    def run_all_checks(self, sample_data=None):
        """Run all validation checks."""
        print("Running production validation checks...")

        model = self.validate_model_loading()

        if model and sample_data is not None:
            self.validate_prediction_format(model, sample_data)

        self.validate_environment()
        self.validate_logging()

        # Print results
        print("\nValidation Results:")
        print("-" * 50)
        for check_name, passed, message in self.checks:
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"{check_name:20} | {status:7} | {message}")

        all_passed = all(check[1] for check in self.checks)
        print(f"\nOverall Status: {'READY' if all_passed else 'NOT READY'}")

        return all_passed

# Usage
validator = ProductionValidator('models/production_model.pkl')
validator.run_all_checks(sample_data=X_test[:5])
```

### Configuration Management

```python
# config.py
import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class DeploymentConfig:
    """Production deployment configuration."""

    # Model settings
    model_path: str = 'models/production_model.pkl'
    model_version: Optional[str] = None

    # API settings
    host: str = '0.0.0.0'
    port: int = 8000
    workers: int = 1

    # Performance settings
    batch_size: int = 1000
    max_request_size: int = 1024 * 1024  # 1MB
    timeout: int = 30

    # Monitoring settings
    enable_metrics: bool = True
    log_level: str = 'INFO'

    # Security settings
    enable_auth: bool = False
    api_key: Optional[str] = None

    @classmethod
    def from_env(cls):
        """Create config from environment variables."""
        return cls(
            model_path=os.getenv('MODEL_PATH', cls.model_path),
            model_version=os.getenv('MODEL_VERSION'),
            host=os.getenv('HOST', cls.host),
            port=int(os.getenv('PORT', cls.port)),
            workers=int(os.getenv('WORKERS', cls.workers)),
            batch_size=int(os.getenv('BATCH_SIZE', cls.batch_size)),
            log_level=os.getenv('LOG_LEVEL', cls.log_level),
            enable_auth=os.getenv('ENABLE_AUTH', 'false').lower() == 'true',
            api_key=os.getenv('API_KEY')
        )

# Load configuration
config = DeploymentConfig.from_env()
```

This deployment guide provides comprehensive strategies for deploying MyML models in various environments, from simple local deployments to scalable cloud solutions with monitoring and A/B testing capabilities.
