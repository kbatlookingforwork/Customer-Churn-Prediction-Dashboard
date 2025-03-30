import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots

def perform_eda(data, column, plot_type='histogram', target_col=None):
    """
    Perform exploratory data analysis on a specific column.
    
    Parameters:
    -----------
    data : pd.DataFrame
        The input dataset
    column : str
        The column to analyze
    plot_type : str, default='histogram'
        Type of plot to generate.
        Options: 'histogram', 'boxplot', 'violinplot', 'countplot'
    target_col : str, optional
        Target column for color differentiation (e.g., 'Churn')
    
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object
    """
    if plot_type == 'histogram':
        if target_col and target_col in data.columns:
            fig = px.histogram(
                data, 
                x=column,
                color=target_col,
                marginal="box",
                opacity=0.7,
                barmode="overlay",
                title=f"Distribution of {column} by {target_col}"
            )
        else:
            fig = px.histogram(
                data, 
                x=column,
                marginal="box",
                title=f"Distribution of {column}"
            )
        
        fig.update_layout(bargap=0.1)
        
    elif plot_type == 'boxplot':
        if target_col and target_col in data.columns:
            fig = px.box(
                data, 
                x=target_col,
                y=column,
                color=target_col,
                title=f"Box Plot of {column} by {target_col}"
            )
        else:
            fig = px.box(
                data, 
                y=column,
                title=f"Box Plot of {column}"
            )
    
    elif plot_type == 'violinplot':
        if target_col and target_col in data.columns:
            fig = px.violin(
                data, 
                x=target_col,
                y=column,
                color=target_col,
                box=True,
                points="all",
                title=f"Violin Plot of {column} by {target_col}"
            )
        else:
            fig = px.violin(
                data, 
                y=column,
                box=True,
                points="all",
                title=f"Violin Plot of {column}"
            )
    
    elif plot_type == 'countplot':
        counts = data[column].value_counts().reset_index()
        counts.columns = [column, 'Count']
        
        fig = px.bar(
            counts, 
            x=column,
            y='Count',
            title=f"Count of {column}",
            text='Count'
        )
        
        fig.update_traces(textposition='outside')
        
        # If we have a target column, create a grouped bar chart
        if target_col and target_col in data.columns:
            crosstab = pd.crosstab(data[column], data[target_col])
            crosstab_pct = pd.crosstab(data[column], data[target_col], normalize='index') * 100
            
            fig = px.bar(
                crosstab, 
                barmode='group',
                title=f"Distribution of {column} by {target_col}"
            )
            
            # Add a second y-axis for percentages
            fig_pct = px.line(
                crosstab_pct[1] if 1 in crosstab_pct.columns else crosstab_pct.iloc[:, 0],
                markers=True,
                title=f"Percentage of {target_col} positive by {column}"
            )
            
            for trace in fig_pct.data:
                trace.yaxis = "y2"
                fig.add_trace(trace)
            
            fig.update_layout(
                yaxis2=dict(
                    title=f"{target_col} Rate (%)",
                    overlaying="y",
                    side="right"
                )
            )
    
    else:
        raise ValueError(f"Unsupported plot type: {plot_type}")
    
    fig.update_layout(
        xaxis_title=column,
        yaxis_title="Count" if plot_type in ['histogram', 'countplot'] else column,
        legend_title=target_col if target_col else None,
        height=500
    )
    
    return fig

def generate_correlation_heatmap(data):
    """
    Generate a correlation heatmap for numerical features.
    
    Parameters:
    -----------
    data : pd.DataFrame
        The input dataset
    
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object with correlation heatmap
    """
    # Select only numerical columns
    numerical_data = data.select_dtypes(include=['int64', 'float64'])
    
    # Calculate correlation matrix
    corr_matrix = numerical_data.corr()
    
    # Create heatmap
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="RdBu_r",
        title="Feature Correlation Heatmap"
    )
    
    fig.update_layout(
        height=700,
        width=700
    )
    
    return fig

def analyze_feature_importance(data, features, target_col):
    """
    Analyze feature importance by comparing distributions across target values.
    
    Parameters:
    -----------
    data : pd.DataFrame
        The input dataset
    features : list
        List of feature names to analyze
    target_col : str
        Target column (e.g., 'Churn')
    
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object with subplots for each feature
    """
    # Check that all features are in the dataframe
    missing_features = [f for f in features if f not in data.columns]
    if missing_features:
        raise ValueError(f"Features not found in data: {missing_features}")
    
    # Check that target column is in the dataframe
    if target_col not in data.columns:
        raise ValueError(f"Target column '{target_col}' not found in data")
    
    # Create a subplot for each feature
    fig = make_subplots(
        rows=len(features), 
        cols=1,
        subplot_titles=[f"Distribution of {f} by {target_col}" for f in features],
        vertical_spacing=0.1
    )
    
    # Add traces for each feature
    for i, feature in enumerate(features):
        row = i + 1
        
        # Get unique target values
        target_values = data[target_col].unique()
        
        for target_value in target_values:
            subset = data[data[target_col] == target_value]
            
            # Add histogram trace
            fig.add_trace(
                go.Histogram(
                    x=subset[feature],
                    name=f"{target_col}={target_value}",
                    opacity=0.7,
                    showlegend=row == 1  # Only show legend for the first row
                ),
                row=row, 
                col=1
            )
    
    # Update layout
    fig.update_layout(
        barmode='overlay',
        height=300 * len(features),
        title_text=f"Feature Distributions by {target_col}",
        legend_title=target_col
    )
    
    # Update x and y axis labels
    for i, feature in enumerate(features):
        row = i + 1
        fig.update_xaxes(title_text=feature, row=row, col=1)
        fig.update_yaxes(title_text="Count", row=row, col=1)
    
    return fig
