"""Command-line interface for sklearn-mastery project."""

import click
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

from .data.generators import SyntheticDataGenerator
from .pipelines.pipeline_factory import PipelineFactory
from .evaluation.metrics import ModelEvaluator
from .evaluation.visualization import ModelVisualizationSuite
from .utils.helpers import DataUtils, ModelUtils
from .config.settings import settings
from .config.logging_config import setup_logging, get_logger


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.option('--log-file', type=str, help='Log file path')
def cli(verbose: bool, log_file: Optional[str]):
    """Scikit-Learn Mastery: Advanced ML Portfolio Tools."""
    log_level = "DEBUG" if verbose else "INFO"
    setup_logging(log_level=log_level, log_file=log_file)
    click.echo("🚀 Scikit-Learn Mastery CLI initialized!")


@cli.command()
@click.option('--dataset-type', type=click.Choice(['regression', 'classification', 'clustering']), 
              default='classification', help='Type of dataset to generate')
@click.option('--n-samples', default=1000, help='Number of samples')
@click.option('--n-features', default=20, help='Number of features')
@click.option('--output', '-o', default='generated_data.csv', help='Output file path')
@click.option('--complexity', type=click.Choice(['linear', 'medium', 'high']), 
              default='medium', help='Dataset complexity level')
def generate_data(dataset_type: str, n_samples: int, n_features: int, 
                 output: str, complexity: str):
    """Generate synthetic datasets optimized for algorithm demonstration."""
    logger = get_logger('cli')
    logger.info(f"Generating {dataset_type} dataset with {n_samples} samples")
    
    generator = SyntheticDataGenerator(random_state=settings.RANDOM_SEED)
    
    if dataset_type == 'regression':
        if complexity == 'linear':
            X, y = generator.linear_regression_data(n_samples, n_features)
        else:
            X, y, _ = generator.regression_with_collinearity(n_samples, n_features)
    
    elif dataset_type == 'classification':
        if n_features == 2:
            X, y = generator.classification_complexity_spectrum(complexity, n_samples)
        else:
            X, y = generator.high_dimensional_sparse_data(n_samples, n_features)
    
    elif dataset_type == 'clustering':
        if complexity == 'linear':
            X = generator.clustering_blobs_with_noise(n_clusters=4, n_samples=n_samples)
            y = None
        else:
            X = generator.clustering_moons(n_samples)
            y = None
    
    # Save data
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if y is not None:
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        df['target'] = y
    else:
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    
    df.to_csv(output_path, index=False)
    
    click.echo(f"✅ Generated {dataset_type} dataset: {X.shape}")
    click.echo(f"💾 Saved to: {output_path}")


@cli.command()
@click.argument('data_file', type=click.Path(exists=True))
@click.option('--algorithm', default='random_forest', 
              help='Algorithm to use (random_forest, svm, logistic_regression, etc.)')
@click.option('--task-type', type=click.Choice(['classification', 'regression']),
              default='classification', help='Type of ML task')
@click.option('--test-size', default=0.2, help='Test set proportion')
@click.option('--cv-folds', default=5, help='Cross-validation folds')
@click.option('--output-dir', '-o', default='results', help='Output directory')
@click.option('--tune-hyperparameters', is_flag=True, help='Enable hyperparameter tuning')
def train_model(data_file: str, algorithm: str, task_type: str, test_size: float,
               cv_folds: int, output_dir: str, tune_hyperparameters: bool):
    """Train and evaluate a model on the provided dataset."""
    logger = get_logger('cli')
    logger.info(f"Training {algorithm} on {data_file}")
    
    # Load data
    df = pd.read_csv(data_file)
    
    if 'target' in df.columns:
        X = df.drop('target', axis=1).values
        y = df['target'].values
    else:
        click.echo("❌ Error: No 'target' column found in dataset")
        return
    
    # Split data
    X_train, X_test, y_train, y_test = DataUtils.smart_train_test_split(
        X, y, test_size=test_size
    )
    
    # Create pipeline
    factory = PipelineFactory(random_state=settings.RANDOM_SEED)
    
    if tune_hyperparameters:
        pipeline = factory.create_pipeline_with_auto_tuning(
            algorithm=algorithm,
            task_type=task_type,
            preprocessing_level='standard'
        )
    else:
        if task_type == 'classification':
            pipeline = factory.create_classification_pipeline(algorithm=algorithm)
        else:
            pipeline = factory.create_regression_pipeline(algorithm=algorithm)
    
    # Train model
    click.echo("🔄 Training model...")
    pipeline.fit(X_train, y_train)
    
    # Evaluate model
    evaluator = ModelEvaluator(task_type=task_type)
    results = evaluator.evaluate_model(
        model=pipeline,
        X_train=X_train, X_test=X_test,
        y_train=y_train, y_test=y_test,
        model_name=algorithm
    )
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path = output_path / f"{algorithm}_model.pkl"
    ModelUtils.save_model(pipeline, model_path, metadata=results)
    
    # Save evaluation results
    results_path = output_path / f"{algorithm}_results.json"
    import json
    with open(results_path, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                serializable_results[key] = value.tolist()
            elif isinstance(value, dict):
                serializable_results[key] = {k: v.tolist() if isinstance(v, np.ndarray) else v 
                                           for k, v in value.items()}
            else:
                serializable_results[key] = value
        json.dump(serializable_results, f, indent=2)
    
    # Print summary
    summary = evaluator.generate_evaluation_summary(results)
    click.echo("\n" + summary)
    
    click.echo(f"\n💾 Model saved to: {model_path}")
    click.echo(f"📊 Results saved to: {results_path}")


@cli.command()
@click.argument('results_dir', type=click.Path(exists=True))
@click.option('--output', '-o', default='comparison_report.html', help='Output HTML report path')
def compare_models(results_dir: str, output: str):
    """Compare multiple model results and generate a report."""
    logger = get_logger('cli')
    logger.info(f"Comparing models in {results_dir}")
    
    results_path = Path(results_dir)
    result_files = list(results_path.glob("*_results.json"))
    
    if not result_files:
        click.echo("❌ No result files found in the specified directory")
        return
    
    # Load all results
    all_results = []
    for result_file in result_files:
        with open(result_file, 'r') as f:
            result = json.load(f)
            all_results.append(result)
    
    # Create comparison
    evaluator = ModelEvaluator()
    comparison_df = evaluator.compare_models(all_results)
    
    # Create visualizations
    viz = ModelVisualizationSuite()
    
    # Determine primary metric
    if all_results[0].get('task_type') == 'classification':
        primary_metric = 'test_accuracy'
    else:
        primary_metric = 'test_r2'
    
    if primary_metric in comparison_df.columns:
        fig = viz.plot_model_comparison(comparison_df, primary_metric)
        
        # Save as HTML
        import plotly.io as pio
        html_content = pio.to_html(fig, include_plotlyjs='cdn')
        
        # Add comparison table
        table_html = comparison_df.to_html(classes='table table-striped', 
                                         table_id='comparison-table')
        
        full_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Model Comparison Report</title>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
        </head>
        <body>
            <div class="container mt-4">
                <h1>Model Comparison Report</h1>
                <h2>Performance Visualization</h2>
                {html_content}
                <h2>Detailed Comparison</h2>
                {table_html}
            </div>
        </body>
        </html>
        """
        
        with open(output, 'w') as f:
            f.write(full_html)
        
        click.echo(f"📊 Comparison report saved to: {output}")
    else:
        click.echo("❌ Could not create comparison - missing performance metrics")


@cli.command()
@click.option('--port', default=8888, help='Port for Jupyter server')
@click.option('--notebook-dir', default='notebooks', help='Notebook directory')
def launch_notebooks(port: int, notebook_dir: str):
    """Launch Jupyter notebooks for interactive exploration."""
    import subprocess
    import os
    
    notebook_path = Path(notebook_dir)
    if not notebook_path.exists():
        click.echo(f"❌ Notebook directory not found: {notebook_path}")
        return
    
    click.echo(f"🚀 Launching Jupyter on port {port}...")
    click.echo(f"📂 Notebook directory: {notebook_path.absolute()}")
    
    try:
        subprocess.run([
            'jupyter', 'lab', 
            '--port', str(port),
            '--notebook-dir', str(notebook_path),
            '--no-browser'
        ], check=True)
    except subprocess.CalledProcessError:
        click.echo("❌ Failed to launch Jupyter. Make sure it's installed:")
        click.echo("pip install jupyterlab")
    except KeyboardInterrupt:
        click.echo("\n👋 Jupyter server stopped")


@cli.command()
def demo():
    """Run a quick demonstration of the sklearn-mastery capabilities."""
    click.echo("🎯 Running sklearn-mastery demonstration...")
    
    # Generate demo data
    generator = SyntheticDataGenerator(random_state=42)
    X, y = generator.classification_complexity_spectrum('medium', n_samples=500)
    
    click.echo(f"✅ Generated demo dataset: {X.shape}")
    
    # Create and train model
    factory = PipelineFactory(random_state=42)
    pipeline = factory.create_classification_pipeline(
        algorithm='random_forest',
        preprocessing_level='standard'
    )
    
    # Split and train
    X_train, X_test, y_train, y_test = DataUtils.smart_train_test_split(X, y)
    
    click.echo("🔄 Training Random Forest model...")
    pipeline.fit(X_train, y_train)
    
    # Evaluate
    evaluator = ModelEvaluator(task_type='classification')
    results = evaluator.evaluate_model(
        pipeline, X_train, X_test, y_train, y_test, "Demo Random Forest"
    )
    
    # Show results
    summary = evaluator.generate_evaluation_summary(results)
    click.echo("\n" + summary)
    
    click.echo("\n🎉 Demo complete! Explore more with:")
    click.echo("  sklearn-mastery generate-data --help")
    click.echo("  sklearn-mastery train-model --help")
    click.echo("  sklearn-mastery launch-notebooks")


def main():
    """Main entry point for CLI."""
    cli()


if __name__ == '__main__':
    main()