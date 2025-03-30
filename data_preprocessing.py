import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def handle_missing_values(data, strategy='mean'):
    """
    Handle missing values in the dataset.
    
    Parameters:
    -----------
    data : pd.DataFrame
        The input dataset
    strategy : str, default='mean'
        Strategy for imputing missing values in numerical columns.
        Options: 'mean', 'median', 'mode', 'drop'
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with missing values handled
    """
    # Create a copy of the data
    df = data.copy()
    
    # If strategy is 'drop', drop rows with missing values
    if strategy == 'drop':
        return df.dropna()
    
    # Get numerical and categorical columns
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    # Handle missing values in numerical columns
    if numerical_cols.size > 0:
        if strategy == 'mode':
            for col in numerical_cols:
                if df[col].isna().sum() > 0:
                    df[col].fillna(df[col].mode()[0], inplace=True)
        else:
            for col in numerical_cols:
                if df[col].isna().sum() > 0:
                    if strategy == 'mean':
                        df[col].fillna(df[col].mean(), inplace=True)
                    elif strategy == 'median':
                        df[col].fillna(df[col].median(), inplace=True)
    
    # Handle missing values in categorical columns (always use mode for categorical)
    if categorical_cols.size > 0:
        for col in categorical_cols:
            if df[col].isna().sum() > 0:
                df[col].fillna(df[col].mode()[0], inplace=True)
    
    return df

def encode_categorical_features(data, encoding_strategy='one-hot'):
    """
    Encode categorical features in the dataset.
    
    Parameters:
    -----------
    data : pd.DataFrame
        The input dataset
    encoding_strategy : str, default='one-hot'
        Strategy for encoding categorical features.
        Options: 'one-hot', 'label'
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with encoded categorical features
    """
    # Create a copy of the data
    df = data.copy()
    
    # Get categorical columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    if categorical_cols.size > 0:
        if encoding_strategy == 'one-hot':
            # Apply one-hot encoding
            df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        elif encoding_strategy == 'label':
            # Apply label encoding
            for col in categorical_cols:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
    
    return df

def scale_numerical_features(data, scaling_strategy='standard'):
    """
    Scale numerical features in the dataset.
    
    Parameters:
    -----------
    data : pd.DataFrame
        The input dataset
    scaling_strategy : str, default='standard'
        Strategy for scaling numerical features.
        Options: 'standard', 'minmax', 'none'
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with scaled numerical features
    """
    # If no scaling needed, return the original data
    if scaling_strategy == 'none':
        return data
    
    # Create a copy of the data
    df = data.copy()
    
    # Get numerical columns
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    
    if numerical_cols.size > 0:
        if scaling_strategy == 'standard':
            # Apply standard scaling
            scaler = StandardScaler()
            df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
        elif scaling_strategy == 'minmax':
            # Apply min-max scaling
            scaler = MinMaxScaler()
            df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    return df

def preprocess_data(data, categorical_encoding='one-hot', scaling='standard'):
    """
    Preprocess the dataset by encoding categorical features and scaling numerical features.
    
    Parameters:
    -----------
    data : pd.DataFrame
        The input dataset
    categorical_encoding : str, default='one-hot'
        Strategy for encoding categorical features.
        Options: 'one-hot', 'label'
    scaling : str, default='standard'
        Strategy for scaling numerical features.
        Options: 'standard', 'minmax', 'none'
    
    Returns:
    --------
    pd.DataFrame
        Preprocessed DataFrame
    """
    # Encode categorical features
    df_encoded = encode_categorical_features(data, encoding_strategy=categorical_encoding)
    
    # Scale numerical features
    df_preprocessed = scale_numerical_features(df_encoded, scaling_strategy=scaling)
    
    return df_preprocessed
