import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, auc
from plotly.subplots import make_subplots

def plot_feature_importance(model, feature_names):
    """
    Plot feature importance for tree-based models.
    
    Parameters:
    -----------
    model : trained model object
        Trained tree-based model (Random Forest, XGBoost)
    feature_names : list
        List of feature names
    
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object
    """
    # Get feature importance
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        # For models that don't have feature_importances_ attribute
        return None
    
    # Create a DataFrame for easier plotting
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })
    
    # Sort by importance
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    # Take top 20 features for readability
    importance_df = importance_df.head(20)
    
    # Create the bar plot
    fig = px.bar(
        importance_df,
        y='Feature',
        x='Importance',
        orientation='h',
        title='Feature Importance',
        labels={'Importance': 'Importance Score', 'Feature': 'Feature Name'},
        color='Importance',
        color_continuous_scale='Viridis'
    )
    
    # Update layout
    fig.update_layout(
        height=600,
        yaxis={'categoryorder': 'total ascending'}
    )
    
    return fig

def plot_churn_probability_distribution(y_true, y_prob):
    """
    Plot ROC curve and churn probability distribution.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_prob : array-like
        Predicted probabilities
    
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object
    """
    # Create a figure with 2 subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("ROC Curve", "Churn Probability Distribution"),
        specs=[[{"type": "xy"}, {"type": "xy"}]]
    )
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    # Add ROC curve to the first subplot
    fig.add_trace(
        go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f'ROC Curve (AUC = {roc_auc:.4f})',
            line=dict(color='darkorange', width=2)
        ),
        row=1, col=1
    )
    
    # Add diagonal line for random classifier
    fig.add_trace(
        go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(color='navy', width=2, dash='dash')
        ),
        row=1, col=1
    )
    
    # Create DataFrame for churn probability distribution
    prob_df = pd.DataFrame({
        'Actual': y_true,
        'Probability': y_prob
    })
    
    # Add histogram for churned customers
    churned = prob_df[prob_df['Actual'] == 1]['Probability']
    fig.add_trace(
        go.Histogram(
            x=churned,
            name='Churned Customers',
            opacity=0.7,
            marker_color='red',
            nbinsx=20
        ),
        row=1, col=2
    )
    
    # Add histogram for non-churned customers
    not_churned = prob_df[prob_df['Actual'] == 0]['Probability']
    fig.add_trace(
        go.Histogram(
            x=not_churned,
            name='Non-Churned Customers',
            opacity=0.7,
            marker_color='green',
            nbinsx=20
        ),
        row=1, col=2
    )
    
    # Update layout
    fig.update_layout(
        height=500,
        width=1000,
        barmode='overlay'
    )
    
    # Update axes
    fig.update_xaxes(title_text="False Positive Rate", range=[0, 1], row=1, col=1)
    fig.update_yaxes(title_text="True Positive Rate", range=[0, 1], row=1, col=1)
    fig.update_xaxes(title_text="Churn Probability", range=[0, 1], row=1, col=2)
    fig.update_yaxes(title_text="Count", row=1, col=2)
    
    return fig

def create_customer_journey_heatmap(data, journey_features):
    """
    Create a heatmap visualization for customer journey analysis.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Dataset containing customer journey information
    journey_features : list
        List of features to include in the journey analysis
    
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object
    """
    # Ensure we have the right columns
    if not all(feature in data.columns for feature in journey_features):
        missing = [f for f in journey_features if f not in data.columns]
        raise ValueError(f"Missing features in data: {missing}")
    
    # Create a copy of the data with relevant features
    journey_data = data[journey_features].copy()
    
    # Bin the features to create a grid for the heatmap
    bins = {}
    binned_data = journey_data.copy()
    
    for feature in journey_features[:-1]:  # Exclude the last feature (usually the target)
        # Determine the number of bins based on feature distribution
        n_bins = min(10, len(data[feature].unique()))
        
        # Create bins and convert intervals to string to avoid JSON serialization issues
        bins[feature] = pd.qcut(
            data[feature],
            q=n_bins,
            duplicates='drop'
        )
        
        # Replace values with bin labels as strings to avoid JSON serialization issues
        binned_data[feature] = bins[feature].astype(str)
    
    # Group by the binned features and aggregate the target (usually churn probability)
    target_col = journey_features[-1]
    
    # Create a pivot table for the heatmap
    if len(journey_features) >= 3:
        # Use the first two features for the heatmap
        pivot_data = binned_data.groupby([journey_features[0], journey_features[1]])[target_col].mean().reset_index()
        
        # Create the heatmap
        fig = px.density_heatmap(
            pivot_data,
            x=journey_features[0],
            y=journey_features[1],
            z=target_col,
            title=f"Customer Journey Heatmap ({target_col} by {journey_features[0]} and {journey_features[1]})",
            labels={journey_features[0]: journey_features[0], journey_features[1]: journey_features[1], target_col: target_col},
            color_continuous_scale='Viridis_r'
        )
    else:
        # If we have only one journey feature, create a bar chart
        summary = binned_data.groupby(journey_features[0])[target_col].mean().reset_index()
        
        fig = px.bar(
            summary,
            x=journey_features[0],
            y=target_col,
            title=f"Customer Journey Analysis ({target_col} by {journey_features[0]})",
            labels={journey_features[0]: journey_features[0], target_col: target_col},
            color=target_col,
            color_continuous_scale='Viridis_r'
        )
    
    # Update layout
    fig.update_layout(
        height=600,
        width=800
    )
    
    return fig

def create_customer_journey_funnel(data, journey_features):
    """
    Create a funnel chart visualization showing customer journey stages and churn risk.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Dataset containing customer journey information
    journey_features : list
        List of features to include in the journey analysis
    
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object with funnel chart
    """
    # Ensure we have the target column (churn probability)
    target_col = journey_features[-1]
    
    # Define customer segments based on tenure or other relevant feature
    if 'tenure' in data.columns:
        # Define customer segments based on tenure
        data['Customer_Stage'] = pd.cut(
            data['tenure'],
            bins=[0, 6, 12, 24, 36, float('inf')],
            labels=['New (0-6 months)', 'Early (6-12 months)', 'Developing (1-2 years)', 
                   'Established (2-3 years)', 'Loyal (3+ years)']
        )
    else:
        # Try to find a suitable column for segmentation
        potential_columns = [col for col in data.columns if any(term in col.lower() 
                                                               for term in ['month', 'year', 'duration', 'stage'])]
        if potential_columns:
            segment_col = potential_columns[0]
            # Create quantile-based segments
            data['Customer_Stage'] = pd.qcut(
                data[segment_col], 
                q=5, 
                labels=['Lowest 20%', 'Low-Mid 20%', 'Middle 20%', 'Mid-High 20%', 'Highest 20%'],
                duplicates='drop'
            ).astype(str)
        else:
            # Default to equal-sized segments based on index
            data['Customer_Stage'] = pd.qcut(
                range(len(data)), 
                q=5, 
                labels=['Group 1', 'Group 2', 'Group 3', 'Group 4', 'Group 5'],
                duplicates='drop'
            ).astype(str)
    
    # Calculate average churn probability and count for each stage
    # Use size() instead of trying to aggregate on index
    stage_counts = data.groupby('Customer_Stage').size().reset_index(name='count')
    stage_metrics = data.groupby('Customer_Stage')[target_col].mean().reset_index(name=target_col)
    
    # Merge the two dataframes
    stage_metrics = pd.merge(stage_counts, stage_metrics, on='Customer_Stage')
    
    # Sort by the natural order of stages
    if 'tenure' in data.columns:
        # Already in correct order from pd.cut
        pass
    else:
        # Sort by count in descending order for the funnel
        stage_metrics.sort_values('count', ascending=False, inplace=True)
    
    # Create funnel chart showing customer count at each stage
    fig = go.Figure()
    
    # Add funnel trace for customer count
    fig.add_trace(go.Funnel(
        name='Customer Count',
        y=stage_metrics['Customer_Stage'],
        x=stage_metrics['count'],
        textposition='inside',
        textinfo='value+percent initial',
        opacity=0.65,
        marker=dict(color='royalblue'),
        connector=dict(line=dict(width=1, color='royalblue'))
    ))
    
    # Update layout
    fig.update_layout(
        title_text="Customer Journey Funnel",
        height=500,
        width=700,
        margin=dict(t=50, l=50, r=50, b=50)
    )
    
    return fig
