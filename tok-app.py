import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
import time
from datetime import datetime

class OptimizedStackedClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.scaler = RobustScaler()
        self.label_encoder = LabelEncoder()
        self.feature_selector = None
        self.base_models = None
        self.meta_model = None
        self.selected_features = None
        self.start_time = time.time()

    def predict(self, X):
        """Make predictions using optimized pipeline"""
        # Scale and select features
        X_scaled = pd.DataFrame(
            self.scaler.transform(X),
            columns=X.columns
        )
        X_selected = X_scaled[self.selected_features]
        
        # Generate meta-features
        meta_features = np.zeros((X_selected.shape[0], len(self.base_models) * 6))
        for i, (name, model) in enumerate(self.base_models):
            predictions = model.predict_proba(X_selected)
            meta_features[:, i*6:(i+1)*6] = predictions
        
        # Make final predictions
        predictions = self.meta_model.predict(meta_features)
        return self.label_encoder.inverse_transform(predictions)

    def predict_proba(self, X):
        """Get prediction probabilities"""
        # Scale and select features
        X_scaled = pd.DataFrame(
            self.scaler.transform(X),
            columns=X.columns
        )
        X_selected = X_scaled[self.selected_features]
        
        # Generate meta-features
        meta_features = np.zeros((X_selected.shape[0], len(self.base_models) * 6))
        for i, (name, model) in enumerate(self.base_models):
            predictions = model.predict_proba(X_selected)
            meta_features[:, i*6:(i+1)*6] = predictions
        
        return self.meta_model.predict_proba(meta_features)

def load_model(model_path):
    """Load the saved model"""
    try:
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def create_features(input_data):
    """Create features matching the model's exact feature names"""
    features = {
        'chars_original': input_data['chars_original'],
        'chars_tokenized': input_data['chars_tokenized'],
        'num_words': input_data['num_words'],
        'num_tokens': input_data['num_tokens'],
        'unique_tokens': input_data['unique_tokens'],
        'type_token_ratio': input_data['type_token_ratio'],
        'fertility': input_data['fertility'],
        'token_std': input_data['token_std'],
        'avg_token_len': input_data['avg_token_len']
    }
    
    # Add derived features
    eps = 1e-10
    features['chars_per_word'] = features['chars_original'] / (features['num_words'] + eps)
    features['chars_per_token'] = features['chars_tokenized'] / (features['num_tokens'] + eps)
    features['tokens_per_word'] = features['num_tokens'] / (features['num_words'] + eps)
    features['token_complexity'] = features['token_std'] * features['avg_token_len']
    features['lexical_density'] = features['unique_tokens'] / (features['num_words'] + eps)
    features['log_chars'] = np.log1p(features['chars_original'])
    features['complexity_score'] = (
        features['token_complexity'] * 
        features['lexical_density'] * 
        features['type_token_ratio']
    )
    
    return pd.DataFrame([features])

def plot_probabilities(probabilities):
    """Create a bar plot of prediction probabilities"""
    fig = go.Figure(data=[
        go.Bar(
            x=[f'Level {i+1}' for i in range(len(probabilities))],
            y=probabilities,
            text=np.round(probabilities, 3),
            textposition='auto'
        )
    ])
    fig.update_layout(
        title='Probability Distribution Across Readability Levels',
        xaxis_title='Readability Level',
        yaxis_title='Probability',
        yaxis_range=[0, 1],
        height=400
    )
    return fig

def plot_feature_values(features_df):
    """Create a bar plot of feature values"""
    fig = go.Figure(data=[
        go.Bar(
            x=features_df.columns,
            y=features_df.values[0],
            text=np.round(features_df.values[0], 2),
            textposition='auto'
        )
    ])
    fig.update_layout(
        title='Feature Values',
        xaxis_title='Features',
        yaxis_title='Value',
        xaxis_tickangle=-45,
        height=500
    )
    return fig

def main():
    st.set_page_config(page_title="Text Readability Classifier", layout="wide")
    
    st.title("Text Readability Classifier")
    st.write("This app predicts the readability level based on text characteristics.")
    
    # Load the model
    model_path = "stacked_classifier_20241124_213512.joblib"
    model = load_model(model_path)
    
    if model is None:
        st.error("Could not load the model. Please check if the model file exists.")
        return
    
    # Create two columns for layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Input form for text characteristics
        st.subheader("Enter Text Characteristics")
        
        # Basic features input
        input_data = {}
        input_data['chars_original'] = st.number_input('Number of Characters (Original)', value=0)
        input_data['chars_tokenized'] = st.number_input('Number of Characters (Tokenized)', value=0)
        input_data['num_words'] = st.number_input('Number of Words', value=0)
        input_data['num_tokens'] = st.number_input('Number of Tokens', value=0)
        input_data['unique_tokens'] = st.number_input('Number of Unique Tokens', value=0)
        input_data['type_token_ratio'] = st.number_input('Type-Token Ratio', value=0.0, min_value=0.0, max_value=1.0)
        input_data['fertility'] = st.number_input('Fertility', value=0.0)
        input_data['token_std'] = st.number_input('Token Standard Deviation', value=0.0)
        input_data['avg_token_len'] = st.number_input('Average Token Length', value=0.0)
        
        analyze_button = st.button("Analyze", type="primary")
        
        if analyze_button:
            with st.spinner("Analyzing..."):
                try:
                    # Create features dataframe with all required features
                    features_df = create_features(input_data)
                    
                    # Make prediction
                    prediction = model.predict(features_df)[0]
                    probabilities = model.predict_proba(features_df)[0]
                    
                    # Display results
                    st.subheader("Analysis Results")
                    
                    # Create metrics row
                    metrics_cols = st.columns(2)
                    with metrics_cols[0]:
                        st.metric("Readability Level", f"Level {prediction}")
                    with metrics_cols[1]:
                        highest_prob = max(probabilities)
                        st.metric("Confidence", f"{highest_prob:.2%}")
                    
                    # Show probability distribution
                    st.plotly_chart(plot_probabilities(probabilities), 
                                  use_container_width=True)
                    
                    # Show all feature values including derived features
                    st.subheader("All Features (Including Derived)")
                    st.plotly_chart(plot_feature_values(features_df),
                                  use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")
    
    with col2:
        # Information sidebar
        with st.container():
            st.subheader("About Readability Levels")
            st.write("""
            The model predicts readability on a scale from 1 to 6:
            - **Level 1**: Very Easy
            - **Level 2**: Easy
            - **Level 3**: Moderately Easy
            - **Level 4**: Moderate
            - **Level 5**: Moderately Difficult
            - **Level 6**: Difficult
            """)
            
            st.subheader("Feature Explanations")
            st.write("""
            **Basic Features:**
            - Character counts (original and tokenized)
            - Word and token counts
            - Type-token ratio (vocabulary diversity)
            - Token length statistics
            
            **Derived Features:**
            - Characters per word/token
            - Token complexity
            - Lexical density
            - Overall complexity score
            """)
            
            st.subheader("Model Performance")
            st.write("""
            This model achieves:
            - **Accuracy**: 73.86%
            - **Macro Avg F1**: 0.75
            - **Weighted Avg F1**: 0.74
            
            *Note: Results should be used as guidance rather than absolute measures.*
            """)

if __name__ == "__main__":
    main()
