import streamlit as st
import pandas as pd
import numpy as np
import time
import io
import plotly.express as px

from data_preprocessing import preprocess_data, encode_categorical_features, handle_missing_values
from exploratory_analysis import perform_eda, generate_correlation_heatmap, analyze_feature_importance
from model_training import train_model, evaluate_model, predict_churn
from visualization import create_customer_journey_heatmap, plot_churn_probability_distribution, plot_feature_importance, create_customer_journey_funnel
from utils import display_metrics, convert_df_to_csv, load_sample_data

# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Application title and description
col1, col2 = st.columns([1, 3])
with col1:
    st.image("assets/businessman-working-on-laptop.png", width=150)
with col2:
    st.title("Customer Churn Prediction Dashboard")
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 10px; margin-top: 20px;">
        <p style="font-weight: bold; color: green;">Created by:</p>
        <a href="https://www.linkedin.com/in/danyyudha" target="_blank">
            <img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" 
                 style="width: 20px; height: 20px;">
        </a>
        <p><b>Dany Yudha Putra Haque</b></p>
    </div>
""", unsafe_allow_html=True)
    st.markdown("### Analyze, Predict, and Reduce Customer Churn")


st.markdown("""
This application helps you predict customer churn using machine learning. 
Upload your customer data to analyze patterns, visualize insights, and identify customers at risk of churning.
""")

# Sidebar for navigation and controls
st.sidebar.title("Navigation")
st.sidebar.image("assets/business-presentation.svg", width=80)
page = st.sidebar.radio("Go to", ["Upload Data", "Data Preprocessing", "Exploratory Analysis", "Model Training & Prediction", "Customer Risk Segmentation"])

# Initialize session state variables if they don't exist
if 'data' not in st.session_state:
    st.session_state.data = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'features' not in st.session_state:
    st.session_state.features = None
if 'target' not in st.session_state:
    st.session_state.target = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'model_metrics' not in st.session_state:
    st.session_state.model_metrics = None

# Upload Data Page
if page == "Upload Data":
    st.header("Upload Your Customer Data")
    st.markdown("""
    Upload a CSV file containing your customer data. The file should include:
    - Customer demographics (age, gender, income, location)
    - Activity data (purchase frequency, total transactions, subscription length)
    - Support history (complaints, support tickets)
    - Churn status (target variable: 1 for churned, 0 for active)
    """)
    
    upload_method = st.radio("Select data source:", ["Upload CSV file", "Use sample data"])
    
    if upload_method == "Upload CSV file":
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            try:
                data = pd.read_csv(uploaded_file)
                st.session_state.data = data
                st.success(f"Data loaded successfully with {data.shape[0]} rows and {data.shape[1]} columns.")
                
                # Display the first few rows of the data
                st.subheader("Preview of uploaded data")
                st.dataframe(data.head())
                
                # Display data information
                st.subheader("Data Information")
                buffer = io.StringIO()
                data.info(buf=buffer)
                s = buffer.getvalue()
                st.text(s)
                
                # Download link for the processed data
                st.download_button(
                    label="Download data as CSV",
                    data=convert_df_to_csv(data),
                    file_name='uploaded_data.csv',
                    mime='text/csv',
                )
                
            except Exception as e:
                st.error(f"Error loading data: {e}")
    
    else:  # Use sample data
        if st.button("Load Sample Data"):
            with st.spinner("Loading sample data..."):
                data = load_sample_data()
                st.session_state.data = data
                st.success(f"Sample data loaded successfully with {data.shape[0]} rows and {data.shape[1]} columns.")
                
                # Display the first few rows of the data
                st.subheader("Preview of sample data")
                st.dataframe(data.head())
                
                # Display data information
                st.subheader("Data Information")
                buffer = io.StringIO()
                data.info(buf=buffer)
                s = buffer.getvalue()
                st.text(s)

# Data Preprocessing Page
elif page == "Data Preprocessing":
    st.header("Data Preprocessing")
    
    if st.session_state.data is None:
        st.warning("Please upload data first on the 'Upload Data' page.")
    else:
        data = st.session_state.data
        
        st.subheader("1. Missing Values Handling")
        st.markdown("Choose how to handle missing values in your dataset:")
        missing_strategy = st.selectbox(
            "Strategy for numerical features:", 
            ["mean", "median", "mode", "drop"], 
            index=0
        )
        
        st.subheader("2. Categorical Feature Encoding")
        st.markdown("Choose how to encode categorical features:")
        encoding_strategy = st.selectbox(
            "Encoding strategy:", 
            ["one-hot", "label"], 
            index=0
        )
        
        st.subheader("3. Feature Scaling")
        st.markdown("Choose if and how to scale numerical features:")
        scaling_strategy = st.selectbox(
            "Scaling strategy:", 
            ["none", "standard", "minmax"], 
            index=1
        )
        
        st.subheader("4. Target Variable")
        st.markdown("Select the column that represents customer churn (target variable):")
        target_options = data.columns.tolist()
        target_column = st.selectbox("Target column:", target_options, index=target_options.index("Churn") if "Churn" in target_options else 0)
        
        if st.button("Preprocess Data"):
            with st.spinner("Preprocessing data..."):
                try:
                    # Handle missing values
                    data_cleaned = handle_missing_values(data, strategy=missing_strategy)
                    
                    # Split features and target
                    X = data_cleaned.drop(columns=[target_column])
                    y = data_cleaned[target_column]
                    
                    # Preprocess the data
                    X_processed = preprocess_data(
                        X, 
                        categorical_encoding=encoding_strategy,
                        scaling=scaling_strategy
                    )
                    
                    # Store the processed data and target in session state
                    st.session_state.processed_data = data_cleaned
                    st.session_state.features = X_processed
                    st.session_state.target = y
                    
                    st.success("Data preprocessing completed successfully!")
                    
                    # Display the processed data
                    st.subheader("Processed Data Preview")
                    st.dataframe(X_processed.head())
                    
                    st.subheader("Feature Information")
                    st.write(f"Number of features after preprocessing: {X_processed.shape[1]}")
                    st.write(f"Target distribution: {y.value_counts().to_dict()}")
                    
                    # Download link for the processed data
                    st.download_button(
                        label="Download processed data as CSV",
                        data=convert_df_to_csv(X_processed),
                        file_name='processed_data.csv',
                        mime='text/csv',
                    )
                    
                except Exception as e:
                    st.error(f"Error during preprocessing: {e}")

# Exploratory Analysis Page
elif page == "Exploratory Analysis":
    st.header("Exploratory Data Analysis")
    
    if st.session_state.processed_data is None:
        st.warning("Please preprocess your data first on the 'Data Preprocessing' page.")
    else:
        data = st.session_state.processed_data
        
        st.subheader("Visualize Data Distributions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
            selected_num_col = st.selectbox("Select a numerical column to visualize:", numerical_cols)
            chart_type = st.selectbox("Chart type:", ["Histogram", "Box Plot", "Violin Plot"])
            
            if chart_type == "Histogram":
                fig = perform_eda(data, selected_num_col, "histogram", target_col="Churn" if "Churn" in data.columns else None)
                st.plotly_chart(fig)
                st.info(f"This histogram shows the distribution of {selected_num_col}. Higher peaks indicate more common values. The box plot on the right shows quartiles and potential outliers.")
            elif chart_type == "Box Plot":
                fig = perform_eda(data, selected_num_col, "boxplot", target_col="Churn" if "Churn" in data.columns else None)
                st.plotly_chart(fig)
                st.info(f"This box plot of {selected_num_col} shows the median (middle line), interquartile range (box), and potential outliers (points). The wider the box, the more variable the data.")
            else:  # Violin Plot
                fig = perform_eda(data, selected_num_col, "violinplot", target_col="Churn" if "Churn" in data.columns else None)
                st.plotly_chart(fig)
                st.info(f"This violin plot shows both the distribution density (width) and statistical summary (box). Where the plot is wider, more data points fall in that range.")
        
        with col2:
            categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
            if categorical_cols:
                selected_cat_col = st.selectbox("Select a categorical column to visualize:", categorical_cols)
                fig = perform_eda(data, selected_cat_col, "countplot", target_col="Churn" if "Churn" in data.columns else None)
                st.plotly_chart(fig)
                st.info(f"This count plot shows the frequency of each category in {selected_cat_col}. Taller bars represent more common categories. If colored by churn, you can see how different categories relate to customer churn.")
            else:
                st.info("No categorical columns found in the dataset.")
        
        st.subheader("Correlation Analysis")
        fig = generate_correlation_heatmap(data)
        st.plotly_chart(fig)
        st.info("This correlation heatmap shows relationships between numerical features. Values close to 1 (dark blue) indicate strong positive correlations, while values close to -1 (dark red) indicate strong negative correlations. You can use this to identify which features are most related to each other.")
        
        st.subheader("Feature Analysis by Churn Status")
        if "Churn" in data.columns:
            num_features = st.multiselect(
                "Select numerical features to analyze:", 
                numerical_cols,
                default=numerical_cols[:3] if len(numerical_cols) >= 3 else numerical_cols
            )
            
            if num_features:
                fig = analyze_feature_importance(data, num_features, "Churn")
                st.plotly_chart(fig)
        else:
            st.info("Target column 'Churn' not found in the dataset.")

# Model Training Page
elif page == "Model Training & Prediction":
    st.header("Model Training & Prediction")
    
    if st.session_state.features is None or st.session_state.target is None:
        st.warning("Please preprocess your data first on the 'Data Preprocessing' page.")
    else:
        X = st.session_state.features
        y = st.session_state.target
        
        st.subheader("Configure Model Training")
        
        col1, col2 = st.columns(2)
        
        with col1:
            model_type = st.selectbox(
                "Select model type:", 
                ["Random Forest", "XGBoost", "Logistic Regression"]
            )
            
            test_size = st.slider(
                "Test set size (%):", 
                min_value=10, 
                max_value=40, 
                value=20
            ) / 100
            
            random_state = st.number_input(
                "Random state (for reproducibility):", 
                min_value=0, 
                max_value=100, 
                value=42
            )
        
        with col2:
            if model_type == "Random Forest":
                n_estimators = st.slider("Number of trees:", 50, 500, 100, 10)
                max_depth = st.slider("Maximum tree depth:", 2, 32, 10, 1)
                min_samples_split = st.slider("Minimum samples to split:", 2, 20, 2, 1)
                model_params = {
                    "n_estimators": n_estimators,
                    "max_depth": max_depth,
                    "min_samples_split": min_samples_split,
                    "random_state": random_state
                }
            
            elif model_type == "XGBoost":
                n_estimators = st.slider("Number of trees:", 50, 500, 100, 10)
                max_depth = st.slider("Maximum tree depth:", 2, 32, 6, 1)
                learning_rate = st.slider("Learning rate:", 0.01, 0.3, 0.1, 0.01)
                model_params = {
                    "n_estimators": n_estimators,
                    "max_depth": max_depth,
                    "learning_rate": learning_rate,
                    "random_state": random_state
                }
            
            else:  # Logistic Regression
                C = st.slider("Regularization strength (C):", 0.01, 10.0, 1.0, 0.01)
                solver = st.selectbox("Solver:", ["liblinear", "lbfgs", "saga"])
                model_params = {
                    "C": C,
                    "solver": solver,
                    "random_state": random_state
                }
        
        if st.button("Train Model"):
            with st.spinner("Training model..."):
                # Train the model
                # Remove random_state from model_params to avoid duplication
                model_params_copy = model_params.copy()
                if "random_state" in model_params_copy:
                    del model_params_copy["random_state"]
                    
                model, X_train, X_test, y_train, y_test = train_model(
                    X, y, 
                    model_type=model_type,
                    test_size=test_size,
                    random_state=random_state,
                    **model_params_copy
                )
                
                # Evaluate the model
                metrics, y_pred, y_prob = evaluate_model(model, X_test, y_test)
                
                # Store model and metrics in session state
                st.session_state.model = model
                st.session_state.model_metrics = metrics
                
                # Make predictions on the entire dataset
                predictions, probabilities = predict_churn(model, X)
                st.session_state.predictions = probabilities
                
                st.success("Model training completed!")
                
                # Display metrics
                st.subheader("Model Performance Metrics")
                display_metrics(metrics)
                
                # Feature importance
                st.subheader("Feature Importance")
                if model_type in ["Random Forest", "XGBoost"]:
                    fig = plot_feature_importance(model, X.columns)
                    st.plotly_chart(fig)
                    st.info("This chart shows which features most influence the model's predictions. Longer bars indicate features with greater impact on predicting customer churn. Focus your retention efforts on improving these key factors.")
                
                # ROC Curve
                st.subheader("ROC Curve")
                fig = plot_churn_probability_distribution(y_test, y_prob)
                st.plotly_chart(fig)
                st.info("The ROC curve shows the model's ability to distinguish between churned and non-churned customers. The higher the curve above the diagonal line and closer to the top-left corner, the better the model performs. The AUC (Area Under the Curve) value closer to 1.0 indicates a stronger model.")

# Customer Risk Segmentation Page
elif page == "Customer Risk Segmentation":
    st.header("Customer Risk Segmentation")
    
    if st.session_state.predictions is None or st.session_state.model is None:
        st.warning("Please train a model first on the 'Model Training & Prediction' page.")
    else:
        data = st.session_state.processed_data.copy()
        data['Churn_Probability'] = st.session_state.predictions
        
        st.subheader("Customer Segmentation by Churn Risk")
        
        # Define risk segments
        data['Risk_Segment'] = pd.cut(
            data['Churn_Probability'], 
            bins=[0, 0.3, 0.7, 1], 
            labels=['Low Risk', 'Medium Risk', 'High Risk']
        )
        
        # Count by segment
        segment_counts = data['Risk_Segment'].value_counts().reset_index()
        segment_counts.columns = ['Risk Segment', 'Count']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Risk Distribution")
            fig = px.pie(
                segment_counts, 
                values='Count', 
                names='Risk Segment',
                color='Risk Segment',
                color_discrete_map={
                    'Low Risk': 'green',
                    'Medium Risk': 'orange',
                    'High Risk': 'red'
                },
                hole=0.4
            )
            st.plotly_chart(fig)
            st.info("This pie chart shows the percentage breakdown of customers by risk level. Red segments represent high-risk customers who are most likely to churn, while green segments show customers with low churn risk.")
        
        with col2:
            st.subheader("Churn Probability Distribution")
            fig = px.histogram(
                data, 
                x='Churn_Probability',
                color='Risk_Segment',
                color_discrete_map={
                    'Low Risk': 'green',
                    'Medium Risk': 'orange',
                    'High Risk': 'red'
                },
                nbins=50
            )
            st.plotly_chart(fig)
            st.info("This histogram shows the distribution of churn probability across all customers. The colors separate customers into risk segments, with red indicating high-risk customers who need immediate retention efforts.")
        
        # Customer journey heatmap
        st.subheader("Customer Journey Analysis")
        
        # Selecting features for customer journey analysis
        if 'TotalCharges' in data.columns and 'tenure' in data.columns:
            journey_features = ['tenure', 'TotalCharges', 'Churn_Probability']
            fig = create_customer_journey_heatmap(data, journey_features)
            st.plotly_chart(fig)
            st.info("This heatmap visualizes how churn risk varies across different customer segments. Darker areas indicate higher churn risk. Use this to identify specific customer segments that need targeted retention strategies based on their characteristics.")
            
            # Add funnel chart for customer journey
            funnel_fig = create_customer_journey_funnel(data, journey_features)
            st.plotly_chart(funnel_fig)
            st.info("This funnel chart shows how customers are distributed across different lifecycle stages. The width of each segment represents the number of customers, helping you identify where customers drop off or become high-risk. Use this visual to prioritize your retention efforts on the most vulnerable segments.")
        else:
            activity_cols = [col for col in data.columns if any(term in col.lower() for term in ['frequency', 'transaction', 'purchase', 'visit'])]
            if activity_cols:
                journey_features = activity_cols[:2] + ['Churn_Probability']
                fig = create_customer_journey_heatmap(data, journey_features)
                st.plotly_chart(fig)
                st.info("This heatmap visualizes how churn risk varies across different customer segments. Darker areas indicate higher churn risk. Use this to identify specific customer segments that need targeted retention strategies based on their characteristics.")
                
                # Add funnel chart for customer journey
                funnel_fig = create_customer_journey_funnel(data, journey_features)
                st.plotly_chart(funnel_fig)
                st.info("This funnel chart shows how customers are distributed across different lifecycle stages. The width of each segment represents the number of customers, helping you identify where customers drop off or become high-risk. Use this visual to prioritize your retention efforts on the most vulnerable segments.")
            else:
                st.info("No suitable features found for customer journey analysis. Consider adding features like 'tenure', 'frequency', or 'transactions'.")
        
        # High-risk customers table
        st.subheader("High Risk Customers")
        high_risk = data[data['Risk_Segment'] == 'High Risk'].sort_values(by='Churn_Probability', ascending=False)
        
        if len(high_risk) > 0:
            display_cols = [col for col in high_risk.columns if col not in ['Risk_Segment']]
            st.dataframe(high_risk[display_cols])
            
            # Download high-risk customer list
            st.download_button(
                label="Download High Risk Customer List",
                data=convert_df_to_csv(high_risk),
                file_name='high_risk_customers.csv',
                mime='text/csv',
            )
        else:
            st.info("No high-risk customers identified.")

# Footer
st.markdown("---")
st.markdown("© 2025 KBATLOOKINGFORWORK | Customer Churn Prediction Dashboard")
