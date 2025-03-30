import streamlit as st
import pandas as pd
import numpy as np
import io
import base64

def display_metrics(metrics):
    """
    Display model evaluation metrics in a nicely formatted way.
    
    Parameters:
    -----------
    metrics : dict
        Dictionary containing model evaluation metrics
    """
    # Create columns for metrics
    col1, col2, col3, col4 = st.columns(4)
    
    # Display key metrics
    col1.metric("Accuracy", f"{metrics['accuracy']:.4f}")
    col2.metric("Precision", f"{metrics['precision']:.4f}")
    col3.metric("Recall", f"{metrics['recall']:.4f}")
    col4.metric("F1 Score", f"{metrics['f1']:.4f}")
    
    # Display AUC-ROC
    st.metric("AUC-ROC", f"{metrics['roc_auc']:.4f}")
    
    # Display confusion matrix
    conf_matrix = metrics['confusion_matrix']
    st.subheader("Confusion Matrix")
    
    # Create confusion matrix heatmap
    import plotly.figure_factory as ff
    
    # Define labels
    categories = ['Not Churned', 'Churned']
    
    # Create annotated heatmap
    fig = ff.create_annotated_heatmap(
        z=conf_matrix,
        x=categories,
        y=categories,
        annotation_text=conf_matrix,
        colorscale='Blues'
    )
    
    # Add text and axes
    fig.update_layout(
        title="Confusion Matrix",
        xaxis=dict(title="Predicted"),
        yaxis=dict(title="Actual"),
        margin=dict(t=50, l=50, r=50, b=50)
    )
    
    # Force the figure to be square
    fig.update_layout(
        width=400,
        height=400
    )
    
    st.plotly_chart(fig)

def convert_df_to_csv(df):
    """
    Convert a dataframe to CSV for download.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe to convert
    
    Returns:
    --------
    str
        CSV string
    """
    return df.to_csv(index=False).encode('utf-8')

def load_sample_data():
    """
    Load sample customer churn data for demonstration.
    
    Returns:
    --------
    pd.DataFrame
        Sample customer churn dataset
    """
    # Create a synthetic dataset for demonstration
    np.random.seed(42)
    # The number of samples to generate, default is 1000
    n_samples = 1000
    
    # Generate customer demographics
    data = {
        'CustomerID': [f'CUST{i:05d}' for i in range(1, n_samples + 1)],
        'Age': np.random.randint(18, 80, n_samples),
        'Gender': np.random.choice(['Male', 'Female'], n_samples),
        'Income': np.random.normal(50000, 20000, n_samples).astype(int),
        'Location': np.random.choice(['Urban', 'Suburban', 'Rural'], n_samples, p=[0.6, 0.3, 0.1])
    }
    
    # Generate customer activity
    data['tenure'] = np.random.randint(1, 72, n_samples)  # Months as customer
    data['MonthlyCharges'] = np.random.uniform(20, 120, n_samples).round(2)
    data['TotalCharges'] = (data['tenure'] * data['MonthlyCharges'] * (0.9 + np.random.random(n_samples) * 0.2)).round(2)
    data['PurchaseFrequency'] = np.random.randint(1, 30, n_samples)
    data['TotalTransactions'] = (data['tenure'] * data['PurchaseFrequency'] / 10).astype(int)
    
    # Generate support history
    data['SupportTickets'] = np.random.poisson(2, n_samples)
    data['Complaints'] = np.random.poisson(0.5, n_samples)
    
    # Additional features
    data['Contract'] = np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples, p=[0.6, 0.3, 0.1])
    data['PaymentMethod'] = np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], n_samples)
    data['OnlineService'] = np.random.choice(['Yes', 'No'], n_samples)
    data['TechSupport'] = np.random.choice(['Yes', 'No'], n_samples)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Generate churn based on features (making it somewhat realistic)
    churn_prob = (
        0.1 +  # base churn rate
        0.3 * (df['Contract'] == 'Month-to-month') +  # higher churn for month-to-month
        0.2 * (df['tenure'] < 12) +  # higher churn for new customers
        0.1 * (df['Complaints'] > 0) +  # higher churn for those with complaints
        0.1 * (df['SupportTickets'] > 3) +  # higher churn for those with many support tickets
        -0.1 * (df['tenure'] > 36) +  # lower churn for long-term customers
        -0.05 * (df['TotalTransactions'] > 100)  # lower churn for frequent purchasers
    )
    
    # Clip probabilities between 0 and 1
    churn_prob = np.clip(churn_prob, 0, 0.9)
    
    # Assign churn status
    df['Churn'] = np.random.random(n_samples) < churn_prob
    df['Churn'] = df['Churn'].astype(int)
    
    return df
