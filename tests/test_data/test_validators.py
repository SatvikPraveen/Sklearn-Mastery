"""
Focused tests for data validation utilities.
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from data.validators import DataValidator, SchemaValidator, ValidationSeverity, ValidationIssue


class TestValidationIssue:
    """Test ValidationIssue dataclass."""
    
    def test_validation_issue_creation(self):
        """Test creating validation issues."""
        issue = ValidationIssue(
            severity=ValidationSeverity.WARNING,
            category="test_category",
            message="Test message",
            column="test_column",
            details={"key": "value"}
        )
        
        assert issue.severity == ValidationSeverity.WARNING
        assert issue.category == "test_category"
        assert issue.message == "Test message"
        assert issue.column == "test_column"
        assert issue.details == {"key": "value"}


class TestDataValidator:
    """Comprehensive tests for DataValidator."""
    
    @pytest.fixture
    def validator(self):
        """Create validator instance."""
        return DataValidator()
    
    @pytest.fixture
    def clean_data(self):
        """Create clean test data."""
        X = pd.DataFrame({
            'num1': np.random.randn(100),
            'num2': np.random.uniform(0, 10, 100),
            'cat1': np.random.choice(['A', 'B', 'C'], 100)
        })
        y = np.random.randint(0, 2, 100)
        return X, y
    
    def test_empty_dataset_validation(self, validator):
        """Test validation of empty dataset."""
        X = pd.DataFrame()
        report = validator.validate_dataset(X)
        
        assert report['validation_status'] == 'FAILED'
        assert any(issue['severity'] == 'critical' for issue in report['detailed_issues'])
    
    def test_single_sample_validation(self, validator):
        """Test validation with very few samples."""
        X = pd.DataFrame({'col1': [1], 'col2': [2]})
        y = np.array([0])
        
        report = validator.validate_dataset(X, y)
        
        # Should have warning about small dataset
        assert any('small dataset' in issue['message'].lower() 
                  for issue in report['detailed_issues'])
    
    def test_mismatched_xy_lengths(self, validator):
        """Test validation with mismatched X and y lengths."""
        X = pd.DataFrame({'col1': range(100)})
        y = np.array(range(50))  # Different length
        
        report = validator.validate_dataset(X, y)
        
        assert report['validation_status'] == 'FAILED'
        assert any('length mismatch' in issue['message'].lower() 
                  for issue in report['detailed_issues'])
    
    def test_missing_values_validation(self, validator):
        """Test missing values detection and severity."""
        # Test different levels of missing data
        test_cases = [
            (0.01, 'info'),      # 1% missing
            (0.1, 'warning'),    # 10% missing  
            (0.3, 'error'),      # 30% missing
            (0.6, 'critical')    # 60% missing
        ]
        
        for missing_fraction, expected_severity in test_cases:
            X = pd.DataFrame({'col1': range(100)})
            n_missing = int(100 * missing_fraction)
            X.loc[:n_missing-1, 'col1'] = np.nan
            
            report = validator.validate_dataset(X)
            
            # Find missing value issues
            missing_issues = [
                issue for issue in report['detailed_issues'] 
                if 'missing' in issue['message'].lower()
            ]
            
            if missing_fraction > 0:
                assert len(missing_issues) > 0
                # Check that severity matches expectation
                severities = [issue['severity'] for issue in missing_issues]
                assert expected_severity in severities
    
    def test_duplicate_detection(self, validator):
        """Test duplicate row detection."""
        X = pd.DataFrame({
            'col1': [1, 2, 3, 1, 2],  # 40% duplicates
            'col2': [1, 2, 3, 1, 2]
        })
        
        report = validator.validate_dataset(X)
        
        duplicate_issues = [
            issue for issue in report['detailed_issues']
            if 'duplicate' in issue['message'].lower()
        ]
        
        assert len(duplicate_issues) > 0
        assert duplicate_issues[0]['details']['duplicate_count'] == 2
    
    def test_constant_features_detection(self, validator):
        """Test constant feature detection."""
        X = pd.DataFrame({
            'constant': [5] * 100,
            'normal': np.random.randn(100)
        })
        
        report = validator.validate_dataset(X)
        
        constant_issues = [
            issue for issue in report['detailed_issues']
            if 'constant' in issue['message'].lower()
        ]
        
        assert len(constant_issues) > 0
        assert 'constant' in constant_issues[0]['details']['constant_features'][0]
    
    def test_infinite_values_detection(self, validator):
        """Test infinite values detection."""
        X = pd.DataFrame({
            'col1': [1.0, 2.0, np.inf, 4.0, -np.inf]
        })
        
        report = validator.validate_dataset(X)
        
        inf_issues = [
            issue for issue in report['detailed_issues']
            if 'infinite' in issue['message'].lower()
        ]
        
        assert len(inf_issues) > 0
        assert inf_issues[0]['details']['infinite_count'] == 2
    
    def test_outlier_detection(self, validator):
        """Test extreme outlier detection."""
        # Create data with known outliers
        normal_data = np.random.randn(95)
        outliers = np.array([100, -100, 150, -150, 200])  # Extreme outliers
        data = np.concatenate([normal_data, outliers])
        
        X = pd.DataFrame({'col1': data})
        
        report = validator.validate_dataset(X)
        
        outlier_issues = [
            issue for issue in report['detailed_issues']
            if 'outlier' in issue['message'].lower()
        ]
        
        assert len(outlier_issues) > 0
    
    def test_high_cardinality_detection(self, validator):
        """Test high cardinality categorical feature detection."""
        # Create high cardinality categorical feature
        high_card_values = [f'category_{i}' for i in range(80)]
        X = pd.DataFrame({
            'high_card': np.random.choice(high_card_values, 100)
        })
        
        report = validator.validate_dataset(X)
        
        cardinality_issues = [
            issue for issue in report['detailed_issues']
            if 'cardinality' in issue['message'].lower()
        ]
        
        assert len(cardinality_issues) > 0
    
    def test_numeric_strings_detection(self, validator):
        """Test detection of numeric data stored as strings."""
        X = pd.DataFrame({
            'numeric_strings': ['1.5', '2.3', '3.7', '4.1', '5.9']
        })
        
        report = validator.validate_dataset(X)
        
        # Should suggest type conversion
        type_issues = [
            issue for issue in report['detailed_issues']
            if 'type conversion' in issue['message'].lower()
        ]
        
        assert len(type_issues) > 0
    
    def test_classification_target_validation(self, validator):
        """Test classification-specific target validation."""
        X = pd.DataFrame({'col1': range(100)})
        
        # Test single class (should fail)
        y_single_class = np.zeros(100)
        report = validator.validate_dataset(X, y_single_class, task_type='classification')
        
        class_issues = [
            issue for issue in report['detailed_issues']
            if 'class' in issue['message'].lower()
        ]
        assert len(class_issues) > 0
        
        # Test severe imbalance
        y_imbalanced = np.array([0] * 95 + [1] * 5)
        report = validator.validate_dataset(X, y_imbalanced, task_type='classification')
        
        imbalance_issues = [
            issue for issue in report['detailed_issues']
            if 'imbalance' in issue['message'].lower()
        ]
        assert len(imbalance_issues) > 0
    
    def test_regression_target_validation(self, validator):
        """Test regression-specific target validation."""
        X = pd.DataFrame({'col1': range(100)})
        
        # Test target with few unique values
        y_few_unique = np.array([1, 2, 3] * 33 + [1])
        report = validator.validate_dataset(X, y_few_unique, task_type='regression')
        
        regression_issues = [
            issue for issue in report['detailed_issues']
            if 'unique values' in issue['message'].lower()
        ]
        assert len(regression_issues) > 0
    
    def test_multicollinearity_detection(self, validator):
        """Test multicollinearity detection."""
        # Create highly correlated features
        base_feature = np.random.randn(100)
        X = pd.DataFrame({
            'feature1': base_feature,
            'feature2': base_feature + np.random.randn(100) * 0.01,  # Highly correlated
            'feature3': np.random.randn(100)  # Independent
        })
        
        report = validator.validate_dataset(X)
        
        correlation_issues = [
            issue for issue in report['detailed_issues']
            if 'multicollinearity' in issue['message'].lower()
        ]
        
        assert len(correlation_issues) > 0
    
    def test_feature_scaling_detection(self, validator):
        """Test feature scaling need detection."""
        X = pd.DataFrame({
            'small_scale': np.random.randn(100) * 0.01,
            'large_scale': np.random.randn(100) * 1000
        })
        
        report = validator.validate_dataset(X)
        
        scaling_issues = [
            issue for issue in report['detailed_issues']
            if 'scale' in issue['message'].lower()
        ]
        
        assert len(scaling_issues) > 0
    
    def test_task_specific_validation(self, validator):
        """Test task-specific requirements validation."""
        X = pd.DataFrame({'col1': range(20)})
        
        # Classification with insufficient samples per class
        y_small_classes = np.array([0] * 2 + [1] * 2 + [2] * 16)
        report = validator.validate_dataset(X, y_small_classes, task_type='classification')
        
        sample_issues = [
            issue for issue in report['detailed_issues']
            if 'insufficient samples' in issue['message'].lower() or 'low samples' in issue['message'].lower()
        ]
        assert len(sample_issues) > 0
    
    def test_clustering_validation(self, validator):
        """Test clustering-specific validation."""
        # All categorical data (not ideal for clustering)
        X = pd.DataFrame({
            'cat1': np.random.choice(['A', 'B'], 100),
            'cat2': np.random.choice(['X', 'Y'], 100)
        })
        
        report = validator.validate_dataset(X, task_type='clustering')
        
        clustering_issues = [
            issue for issue in report['detailed_issues']
            if 'clustering' in issue['message'].lower() or 'numerical' in issue['message'].lower()
        ]
        
        assert len(clustering_issues) > 0
    
    def test_validation_status_determination(self, validator):
        """Test validation status determination logic."""
        # Test PASSED status
        X = pd.DataFrame({'col1': np.random.randn(100)})
        y = np.random.randint(0, 2, 100)
        report = validator.validate_dataset(X, y)
        assert report['validation_status'] in ['PASSED', 'PASSED_WITH_WARNINGS']
        
        # Test FAILED status (critical issues)
        X_empty = pd.DataFrame()
        report = validator.validate_dataset(X_empty)
        assert report['validation_status'] == 'FAILED'
    
    def test_recommendations_generation(self, validator):
        """Test that recommendations are generated appropriately."""
        X = pd.DataFrame({
            'missing_col': [1, 2, np.nan, 4, 5],
            'constant_col': [1] * 5,
            'outlier_col': [1, 2, 3, 100, 5]
        })
        
        report = validator.validate_dataset(X)
        recommendations = report['recommendations']
        
        assert len(recommendations) > 0
        assert any('data cleaning' in rec.lower() for rec in recommendations)
    
    def test_custom_validation_rules(self, validator):
        """Test adding custom validation rules."""
        def custom_rule(X, y):
            issues = []
            if 'special_col' not in X.columns:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category='custom',
                    message='Missing special column'
                ))
            return issues
        
        validator.add_custom_rule('special_column_check', custom_rule)
        
        X = pd.DataFrame({'normal_col': range(10)})
        # Note: This test demonstrates the interface, but the current implementation
        # doesn't automatically execute custom rules in validate_dataset
    
    def test_strict_mode(self):
        """Test strict mode behavior."""
        validator_strict = DataValidator(strict_mode=True)
        validator_normal = DataValidator(strict_mode=False)
        
        # Both should have the same validation logic in current implementation
        # This test verifies the interface exists
        assert validator_strict.strict_mode is True
        assert validator_normal.strict_mode is False
    
    def test_validation_report_structure(self, validator):
        """Test that validation report has expected structure."""
        X = pd.DataFrame({'col1': range(100)})
        y = np.random.randint(0, 2, 100)
        
        report = validator.validate_dataset(X, y, task_type='classification')
        
        # Check required fields
        required_fields = [
            'validation_status', 'severity_counts', 'total_issues',
            'dataset_statistics', 'quality_metrics', 'issues_by_category',
            'recommendations', 'detailed_issues'
        ]
        
        for field in required_fields:
            assert field in report
        
        # Check structure of nested fields
        assert isinstance(report['severity_counts'], dict)
        assert isinstance(report['dataset_statistics'], dict)
        assert isinstance(report['quality_metrics'], dict)
        assert isinstance(report['detailed_issues'], list)
        assert isinstance(report['recommendations'], list)
    
    def test_print_summary_output(self, validator, capsys):
        """Test print_summary output formatting."""
        X = pd.DataFrame({
            'col1': [1, 2, np.nan, 4, 5],  # Missing values
            'col2': [np.inf, 2, 3, 4, 5]   # Infinite values
        })
        
        validator.validate_dataset(X)
        validator.print_summary()
        
        captured = capsys.readouterr()
        
        # Should contain expected sections
        assert "Validation Summary" in captured.out
        assert "issues found" in captured.out
        
        # Should contain severity indicators
        severity_indicators = ['🔴', '❌', '⚠️', 'ℹ️']
        assert any(indicator in captured.out for indicator in severity_indicators)


class TestSchemaValidator:
    """Test SchemaValidator class."""
    
    def test_required_columns_validation(self):
        """Test required columns validation."""
        schema = {
            'required_columns': ['col1', 'col2', 'col3']
        }
        
        validator = SchemaValidator(schema)
        
        # Test with all required columns
        X_complete = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [4, 5, 6],
            'col3': [7, 8, 9]
        })
        
        assert validator.validate(X_complete) is True
        assert len(validator.get_validation_errors()) == 0
        
        # Test with missing columns
        X_incomplete = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [4, 5, 6]
            # Missing col3
        })
        
        assert validator.validate(X_incomplete) is False
        errors = validator.get_validation_errors()
        assert len(errors) > 0
        assert 'col3' in errors[0]
    
    def test_column_types_validation(self):
        """Test column types validation."""
        schema = {
            'column_types': {
                'int_col': 'int',
                'float_col': 'float',
                'obj_col': 'object'
            }
        }
        
        validator = SchemaValidator(schema)
        
        # Test with correct types
        X_correct = pd.DataFrame({
            'int_col': [1, 2, 3],
            'float_col': [1.1, 2.2, 3.3],
            'obj_col': ['a', 'b', 'c']
        })
        
        assert validator.validate(X_correct) is True
        
        # Test with incorrect types
        X_incorrect = pd.DataFrame({
            'int_col': [1.1, 2.2, 3.3],  # Should be int
            'float_col': [1.1, 2.2, 3.3],
            'obj_col': ['a', 'b', 'c']
        })
        
        assert validator.validate(X_incorrect) is False
        errors = validator.get_validation_errors()
        assert any('int_col' in error for error in errors)
    
    def test_value_ranges_validation(self):
        """Test value ranges validation."""
        schema = {
            'value_ranges': {
                'bounded_col': (0, 10),
                'positive_col': (0, float('inf'))
            }
        }
        
        validator = SchemaValidator(schema)
        
        # Test with values in range
        X_valid = pd.DataFrame({
            'bounded_col': [1, 5, 9],
            'positive_col': [1, 100, 1000]
        })
        
        assert validator.validate(X_valid) is True
        
        # Test with values out of range
        X_invalid = pd.DataFrame({
            'bounded_col': [1, 5, 15],  # 15 > 10
            'positive_col': [1, -5, 1000]  # -5 < 0
        })
        
        assert validator.validate(X_invalid) is False
        errors = validator.get_validation_errors()
        assert len(errors) >= 2  # Should have errors for both columns
    
    def test_complex_schema_validation(self):
        """Test validation with complex schema."""
        schema = {
            'required_columns': ['id', 'value', 'category'],
            'column_types': {
                'id': 'int',
                'value': 'float',
                'category': 'object'
            },
            'value_ranges': {
                'id': (1, 1000),
                'value': (0, 100)
            }
        }
        
        validator = SchemaValidator(schema)
        
        # Test completely valid data
        X_valid = pd.DataFrame({
            'id': [1, 2, 3],
            'value': [10.5, 20.3, 30.7],
            'category': ['A', 'B', 'C']
        })
        
        assert validator.validate(X_valid) is True
        assert len(validator.get_validation_errors()) == 0
        
        # Test data with multiple violations
        X_invalid = pd.DataFrame({
            'id': [1, 2, 3],
            'value': [10.5, 150.0, 30.7],  # 150.0 > 100
            'category': ['A', 'B', 'C']
            # All required columns present, types correct, but value out of range
        })
        
        assert validator.validate(X_invalid) is False
        errors = validator.get_validation_errors()
        assert len(errors) == 1  # Should have one range error
        assert 'value' in errors[0]
    
    def test_validation_errors_cleared(self):
        """Test that validation errors are cleared between validations."""
        schema = {'required_columns': ['col1']}
        validator = SchemaValidator(schema)
        
        # First validation with missing column
        X1 = pd.DataFrame({'col2': [1, 2, 3]})
        validator.validate(X1)
        errors1 = validator.get_validation_errors()
        assert len(errors1) > 0
        
        # Second validation with correct data
        X2 = pd.DataFrame({'col1': [1, 2, 3]})
        validator.validate(X2)
        errors2 = validator.get_validation_errors()
        assert len(errors2) == 0  # Errors should be cleared
    
    def test_partial_schema_validation(self):
        """Test validation with partial schema (only some constraints)."""
        # Schema with only column types, no required columns or ranges
        schema = {
            'column_types': {
                'existing_col': 'int'
            }
        }
        
        validator = SchemaValidator(schema)
        
        # DataFrame with additional columns not in schema
        X = pd.DataFrame({
            'existing_col': [1, 2, 3],
            'extra_col': ['a', 'b', 'c']
        })
        
        # Should pass since we only validate what's in the schema
        assert validator.validate(X) is True
    
    def test_empty_schema_validation(self):
        """Test validation with empty schema."""
        validator = SchemaValidator({})
        
        X = pd.DataFrame({'any_col': [1, 2, 3]})
        
        # Empty schema should always pass
        assert validator.validate(X) is True
        assert len(validator.get_validation_errors()) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])