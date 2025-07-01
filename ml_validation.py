from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score
import numpy as np
from logging_config import setup_logging
import pandas as pd

# Set up logging
ml_logger, _ = setup_logging()

def validate_predictions(predictions, actual_outcomes):
    """
    Validate model predictions against actual outcomes
    Returns metrics and identifies potential issues
    """
    try:
        predictions = np.array(predictions)
        actual_outcomes = np.array(actual_outcomes)
        
        # Calculate metrics
        auc_score = roc_auc_score(actual_outcomes, predictions)
        precision, recall, _ = precision_recall_curve(actual_outcomes, predictions)
        avg_precision = average_precision_score(actual_outcomes, predictions)
        
        # Check for prediction bias
        mean_prediction = np.mean(predictions)
        mean_outcome = np.mean(actual_outcomes)
        prediction_bias = mean_prediction - mean_outcome
        
        # Check for calibration
        calibration_error = np.mean(np.abs(predictions - actual_outcomes))
        
        # Log results
        ml_logger.info(f"Model Validation Results:")
        ml_logger.info(f"AUC Score: {auc_score:.3f}")
        ml_logger.info(f"Average Precision: {avg_precision:.3f}")
        ml_logger.info(f"Prediction Bias: {prediction_bias:.3f}")
        ml_logger.info(f"Calibration Error: {calibration_error:.3f}")
        
        # Flag potential issues
        if abs(prediction_bias) > 0.1:
            ml_logger.warning(f"High prediction bias detected: {prediction_bias:.3f}")
        if calibration_error > 0.2:
            ml_logger.warning(f"High calibration error detected: {calibration_error:.3f}")
        
        return {
            'auc_score': auc_score,
            'avg_precision': avg_precision,
            'prediction_bias': prediction_bias,
            'calibration_error': calibration_error
        }
        
    except Exception as e:
        ml_logger.error(f"Error in prediction validation: {str(e)}")
        return None

def analyze_feature_importance(feature_importance_dict):
    """
    Analyze feature importance and provide insights
    """
    try:
        # Convert to DataFrame for easier analysis
        fi_df = pd.DataFrame({
            'feature': list(feature_importance_dict.keys()),
            'importance': list(feature_importance_dict.values())
        })
        
        # Sort by importance
        fi_df = fi_df.sort_values('importance', ascending=False)
        
        # Log top features
        ml_logger.info("Top 5 most important features:")
        for _, row in fi_df.head().iterrows():
            ml_logger.info(f"{row['feature']}: {row['importance']:.3f}")
        
        # Check for potential issues
        if fi_df['importance'].max() > 0.5:
            ml_logger.warning("High feature dominance detected - model may be overly dependent on a single feature")
        
        if fi_df['importance'].std() < 0.01:
            ml_logger.warning("Low feature importance variation - features may not be sufficiently discriminative")
        
        return fi_df
        
    except Exception as e:
        ml_logger.error(f"Error in feature importance analysis: {str(e)}")
        return None

def validate_feature_collection(features_df):
    """
    Validate collected features for completeness and quality
    """
    try:
        # Check for missing values
        missing_cols = features_df.columns[features_df.isnull().any()].tolist()
        if missing_cols:
            ml_logger.warning(f"Missing values detected in columns: {missing_cols}")
        
        # Check for zero variance features
        zero_var_cols = features_df.columns[features_df.std() == 0].tolist()
        if zero_var_cols:
            ml_logger.warning(f"Zero variance detected in columns: {zero_var_cols}")
        
        # Check for highly correlated features
        correlation_matrix = features_df.corr()
        high_correlation_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i):
                if abs(correlation_matrix.iloc[i, j]) > 0.9:
                    high_correlation_pairs.append(
                        (correlation_matrix.columns[i], 
                         correlation_matrix.columns[j], 
                         correlation_matrix.iloc[i, j])
                    )
        
        if high_correlation_pairs:
            ml_logger.warning("Highly correlated features detected:")
            for feat1, feat2, corr in high_correlation_pairs:
                ml_logger.warning(f"{feat1} - {feat2}: {corr:.3f}")
        
        return {
            'missing_columns': missing_cols,
            'zero_variance_columns': zero_var_cols,
            'high_correlation_pairs': high_correlation_pairs
        }
        
    except Exception as e:
        ml_logger.error(f"Error in feature validation: {str(e)}")
        return None
