import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from keras.models import load_model

# Define the focal loss function used during training
def binary_focal_loss(gamma=2., alpha=0.6):
    def focal_loss_fixed(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.keras.backend.clip(y_pred, epsilon, 1. - epsilon)

        p_t = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        alpha_t = tf.where(tf.equal(y_true, 1), alpha, 1 - alpha)

        loss = -alpha_t * tf.keras.backend.pow(1. - p_t, gamma) * tf.keras.backend.log(p_t)
        return tf.keras.backend.mean(loss)
    return focal_loss_fixed

def treat_missing_values(group, day_cols):
    """Handle missing values based on feature type"""
    feature = group['FEATURE'].iloc[0]

    if feature in ['TOTAL_REVENUE_WT', 'TOTAL_VOICE_USAGE', 'MBB_TOTAL_USAGE', 'PAYG_MB_RATIO']:
        # Fill NaNs with 0.0
        group[day_cols] = group[day_cols].fillna(0.0)
    elif feature in ['WALLET_BALANCE', 'REMAINING_DATA_BAL_MB']:
        # Interpolate across the 30 days (row-wise), fill any remaining NaNs
        group[day_cols] = group[day_cols].interpolate(axis=1, limit_direction='both').fillna(0.0)

    return group

def preprocess_data(df):
    """Preprocess the input data for model inference"""
    # Define day columns
    day_cols = [f'DAY{i}' for i in range(1, 31)]
    
    # Handle missing values
    # Step 1: Replace string "\N" with np.nan
    df[day_cols] = df[day_cols].replace('\\N', np.nan)
    
    # Step 2: Convert to numeric and coerce non-convertibles to NaN
    df[day_cols] = df[day_cols].apply(pd.to_numeric, errors='coerce')
    
    # Apply per-feature missing value treatment
    df = df.groupby('FEATURE', group_keys=False).apply(lambda x: treat_missing_values(x, day_cols))
    
    # Define feature groups
    feature_groups = [
        "TOTAL_REVENUE_WT",
        "TOTAL_VOICE_USAGE",
        "WALLET_BALANCE",
        "MBB_TOTAL_USAGE",
        "PAYG_MB_RATIO",
        "REMAINING_DATA_BAL_MB"
    ]
    
    # Split data by feature
    feature_map = {}
    for feature in feature_groups:
        feature_map[feature] = df[df['FEATURE'] == feature].drop(columns=['MSISDN_ENCR_INT', 'FEATURE'])
    
    # Process each feature group
    processed_inputs = []
    for feature, data in feature_map.items():
        if data is not None and not data.empty:
            # Extract day columns
            X = data[day_cols].values
            
            # Scale the data
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Reshape for BiLSTM: (samples, timesteps, features)
            X_reshaped = np.reshape(X_scaled, (X_scaled.shape[0], X_scaled.shape[1], 1))
            processed_inputs.append(X_reshaped)
    
    return processed_inputs

# Streamlit app code
def run_app():
    st.title('Churn Prediction App with Summary Statistics')

    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    # Load the main data
    main_data_path = 'data/Inference_data.csv'
    main_df = pd.read_csv(main_data_path)

    if uploaded_file is not None:
        # Read the uploaded CSV file
        uploaded_df  = pd.read_csv(uploaded_file)

        # Remove duplicate records based on MSISDN_ENCR_INT
        uploaded_df = uploaded_df.drop_duplicates(subset='MSISDN_ENCR_INT', keep='first')

        # Join the uploaded data with the main data based on MSISDN_ENCR_INT
        df = uploaded_df.merge(main_df, on='MSISDN_ENCR_INT', how='inner')

        # Preprocess the data
        processed_inputs = preprocess_data(df)

        # Instantiate the focal loss function
        focal_loss_fn = binary_focal_loss(gamma=2, alpha=0.6)

        # Load the model with custom loss function
        model = load_model('models/best_model_final.keras', custom_objects={'focal_loss_fixed': focal_loss_fn})

        # Make predictions
        predictions_prob = model.predict(processed_inputs)
            
        # Apply optimal threshold (from training)
        optimal_threshold = 0.47  # This was determined during model training
        predictions = (predictions_prob > optimal_threshold).astype(int)
        
        # Create a DataFrame with results
        results_df = pd.DataFrame({
            'MSISDN_ENCR_INT': df['MSISDN_ENCR_INT'].unique(),
            'Churn_Probability': predictions_prob.flatten(),
            'Predicted_Churn': predictions.flatten()
        })

        # Calculate the required counts
        input_count = len(uploaded_df)
        output_count = len(results_df)

        # Calculate the number of duplicate records based on MSISDN_ENCR_INT
        duplicate_count = uploaded_df.duplicated(subset='MSISDN_ENCR_INT').sum()

        drop_number_count = input_count - output_count - duplicate_count

        churn_count = results_df['Predicted_Churn'].sum()
        non_churn_count = len(results_df) - churn_count

        # Create a DataFrame to display the counts in a single row
        counts_df = pd.DataFrame({
            "Input Count": [input_count],
            "Output Count": [output_count],
            "Duplicate Count": [duplicate_count],
            "Drop Number Count": [drop_number_count],
            "Churn Count": [churn_count],
            "Non-Churn Count": [non_churn_count]
        })

        st.write("### Data Processing Log")
        st.write(counts_df)

        # Header for Summary Stats
        st.write("### Summary Statistics")

        # Create two columns for parallel display
        col1, col2 = st.columns(2)

        # Cluster-wise customer count
        cluster_counts = results_df['Predicted_Churn'].value_counts().sort_index()

        # Cluster-wise churn rate
        churn_rate = results_df['Predicted_Churn'].value_counts().sort_index()

        # Displaying Cluster-wise Customer Count as a Pie Chart
        with col1:
          st.write("#### Cluster wise Customer Count")
          fig1, ax1 = plt.subplots()
          ax1.pie(cluster_counts, labels=cluster_counts.index, autopct='%1.1f%%', startangle=90)
          ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
          st.pyplot(fig1)

        # Displaying Cluster-wise Churn Rate
        with col2:
          st.write("#### Cluster wise Churn Rate")
          fig2, ax2 = plt.subplots()
          ax2.bar(churn_rate.index, churn_rate.values)
          ax2.set_xlabel("Cluster")
          ax2.set_ylabel("Churn Rate")
          ax2.set_title("Cluster wise Churn Rate")
          st.pyplot(fig2)

        st.write("### Predictions:")
        st.write(results_df.head())

        # Download the result as a CSV file
        st.download_button(
            label="Download Predictions as CSV",
            data=results_df.to_csv(index=False).encode('utf-8'),
            file_name='predictions.csv',
            mime='text/csv'
        )

# Run the Streamlit app
if __name__ == "__main__":
    run_app()