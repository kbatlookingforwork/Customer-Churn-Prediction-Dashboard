import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

def train_model(X, y, model_type='Random Forest', test_size=0.2, random_state=42, **model_params):
    """
    Train a machine learning model for churn prediction.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target variable
    model_type : str, default='Random Forest'
        Type of model to train.
        Options: 'Random Forest', 'XGBoost', 'Logistic Regression'
    test_size : float, default=0.2
        Proportion of the dataset to include in the test split
    random_state : int, default=42
        Random seed for reproducibility
    **model_params : dict
        Additional parameters to pass to the model
    
    Returns:
    --------
    model : trained model object
    X_train : pd.DataFrame
        Training features
    X_test : pd.DataFrame
        Testing features
    y_train : pd.Series
        Training target
    y_test : pd.Series
        Testing target
    """
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Initialize the model
    if model_type == 'Random Forest':
        model = RandomForestClassifier(**model_params)
    elif model_type == 'XGBoost':
        model = xgb.XGBClassifier(**model_params)
    elif model_type == 'Logistic Regression':
        model = LogisticRegression(**model_params)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Train the model
    model.fit(X_train, y_train)
    
    return model, X_train, X_test, y_train, y_test

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model.
    
    Parameters:
    -----------
    model : trained model object
    X_test : pd.DataFrame
        Testing features
    y_test : pd.Series
        Testing target
    
    Returns:
    --------
    metrics : dict
        Dictionary of evaluation metrics
    y_pred : np.array
        Predicted labels
    y_prob : np.array
        Predicted probabilities
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Get prediction probabilities
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test)
        # Check if we have binary classification (2 classes) or just 1 class
        if proba.shape[1] > 1:
            y_prob = proba[:, 1]  # For binary classification, take probability of positive class
        else:
            # If only one class is present, set probabilities to match that class
            if np.all(y_pred == 1):  # If the only class is positive
                y_prob = np.ones(len(y_test))
            else:
                y_prob = np.zeros(len(y_test))
    else:
        # If model doesn't have predict_proba, use decision_function if available
        if hasattr(model, "decision_function"):
            y_prob = model.decision_function(X_test)
        else:
            # Fallback to raw predictions
            y_prob = y_pred.astype(float)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else 0,
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
    }
    
    return metrics, y_pred, y_prob

def predict_churn(model, X):
    """
    Make churn predictions on new data.
    
    Parameters:
    -----------
    model : trained model object
    X : pd.DataFrame
        Features to predict on
    
    Returns:
    --------
    predictions : np.array
        Predicted labels
    probabilities : np.array
        Predicted probabilities of churn
    """
    # Make predictions
    predictions = model.predict(X)
    
    # Get prediction probabilities
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        # Check if we have binary classification (2 classes) or just 1 class
        if proba.shape[1] > 1:
            probabilities = proba[:, 1]  # For binary classification, take probability of positive class
        else:
            # If only one class is present, set probabilities to match that class
            if np.all(predictions == 1):  # If the only class is positive
                probabilities = np.ones(len(predictions))
            else:
                probabilities = np.zeros(len(predictions))
    else:
        # If model doesn't have predict_proba, use decision_function if available
        if hasattr(model, "decision_function"):
            probabilities = model.decision_function(X)
        else:
            # Fallback to raw predictions
            probabilities = predictions.astype(float)
    
    return predictions, probabilities
