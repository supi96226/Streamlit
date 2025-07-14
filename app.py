import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Function to cap outliers in a DataFrame
def cap_outliers(df, whisker_width=1.5):
    for column in df.columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - (whisker_width * IQR)
        upper_bound = Q3 + (whisker_width * IQR)
        df[column] = np.where(df[column] < lower_bound, lower_bound, df[column])
        df[column] = np.where(df[column] > upper_bound, upper_bound, df[column])
    return df

# Streamlit app code
def run_app():
    st.title('Churn Prediction App with Summary Statistics')

    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    # Load the main data
    main_data_path = 'data/synthetic_inference_data.csv'
    main_df = pd.read_csv(main_data_path)

    if uploaded_file is not None:
        # Read the uploaded CSV file
        uploaded_df  = pd.read_csv(uploaded_file)

        # Remove duplicate records based on MSISDN_ENCR_INT
        uploaded_df = uploaded_df.drop_duplicates(subset='MSISDN_ENCR_INT', keep='first')

        # Join the uploaded data with the main data based on MSISDN_ENCR_INT
        df = uploaded_df.merge(main_df, on='MSISDN_ENCR_INT', how='inner')

        # Preprocessing steps
        df_pre = df.drop(columns=['MSISDN_ENCR_INT'])
        df_capped = cap_outliers(df_pre)
        scaler = StandardScaler()
        scaled_array = scaler.fit_transform(df_capped)
        df_scaled = pd.DataFrame(scaled_array, index=df_capped.index, columns=df_capped.columns)
        df_combined = pd.concat([df[['MSISDN_ENCR_INT']], df_scaled], axis=1)

        # Predict the cluster ID using a pre-trained Decision Tree model
        dt_model = joblib.load('models/dt_cluster_model.pkl')
        predicted_clusters = dt_model.predict(df_combined[['AON', 'LM_ARPU']])
        df_combined['CLUSTER'] = predicted_clusters

        # Prepare df_final with a copy of the original df
        df_final = df.copy()
        df_final['CLUSTER'] = pd.NA
        df_final['PREDICTION'] = pd.NA

        df_final.loc[df_final['MSISDN_ENCR_INT'].isin(df_combined['MSISDN_ENCR_INT']), 'CLUSTER'] = predicted_clusters

        # Total number of clusters
        num_clusters = 3

        # Iterate through each cluster
        for i in range(num_clusters):
            # Filter the data for the current cluster
            df_clus = df_combined[df_combined['CLUSTER'] == i].drop(columns=['AON', 'LM_ARPU'])

            # Construct the model filename dynamically based on the cluster number
            model_file = f'models/seg_model_c{i}.pkl'

            # Load the model for the current cluster
            loaded_model = joblib.load(model_file)

            # Make predictions on the test set
            y_pred = loaded_model.predict(df_clus.drop(columns='MSISDN_ENCR_INT'))

            # Assign predictions to the corresponding rows in df_final
            df_final.loc[df_final['MSISDN_ENCR_INT'].isin(df_clus['MSISDN_ENCR_INT']), 'PREDICTION'] = y_pred

        df_down = df_final[['MSISDN_ENCR_INT', 'CLUSTER', 'PREDICTION']]

        # Calculate the required counts
        input_count = len(uploaded_df)
        output_count = len(df_down)

        # Calculate the number of duplicate records based on MSISDN_ENCR_INT
        duplicate_count = uploaded_df.duplicated(subset='MSISDN_ENCR_INT').sum()

        drop_number_count = input_count - output_count - duplicate_count

        churn_count = df_final['PREDICTION'].sum()
        non_churn_count = len(df_final) - churn_count

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
        cluster_counts = df_down['CLUSTER'].value_counts().sort_index()

        # Cluster-wise churn rate
        churn_rate = df_down.groupby('CLUSTER')['PREDICTION'].mean().sort_index()

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
        st.write(df_down.head())

        # Download the result as a CSV file
        st.download_button(
            label="Download Predictions as CSV",
            data=df_down.to_csv(index=False).encode('utf-8'),
            file_name='predictions.csv',
            mime='text/csv'
        )

# Run the Streamlit app
if __name__ == "__main__":
    run_app()