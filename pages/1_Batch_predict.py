import streamlit as st
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils.db_utils import extract_prediction_data
from utils.gui_utils import section_load_data_element, section_select_model_and_threshold, section_run_prediction


def section_plot_churn_distribution(result_df):
    """
    Plot the distribution of churn predictions.

    :param result_df: DataFrame containing the churn prediction results.
    """

    fig, ax = plt.subplots()
    sns.countplot(x="Predicted_Churn", data=result_df, ax=ax, order=[0, 1])
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f'{int(height)}', (p.get_x() + p.get_width() / 2., height), ha='center', va='bottom')
    ax.set_title("Predicted Churn Distribution")
    ax.set_xlabel("Prediction Label")
    ax.set_ylabel("Count")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["No Churn", "Churn"])
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    _, pcd_col2, _ = st.columns([1, 3, 1])
    with pcd_col2:
        st.pyplot(fig, use_container_width=True)
                  

def section_feature_wise_analysis(df, result_df):
    """
    Perform feature-wise analysis on the DataFrame and result DataFrame.
    This function will generate plots for each feature based on churn predictions.

    :param df: DataFrame containing the raw data.
    :param result_df: DataFrame containing the churn prediction results.
    """

    feature_options = [
        "TOTAL_REVENUE_WT",
        "TOTAL_VOICE_USAGE",
        "WALLET_BALANCE",
        "MBB_TOTAL_USAGE",
        "PAYG_MB_RATIO",
        "REMAINING_DATA_BAL_MB"
    ]
    # selected_features = [feature for feature in feature_options if st.checkbox(f"Show {feature}", key=feature)]
    selected_features = []
    _, sf_col2, sf_col3, _ = st.columns([1, 3, 3, 1])
    for idx, feature in enumerate(feature_options):
        if idx < 3:
            with sf_col2:
                if st.checkbox(f"Show {feature}", key=feature):
                    selected_features.append(feature)
        else:
            with sf_col3:
                if st.checkbox(f"Show {feature}", key=feature):
                    selected_features.append(feature)

    for feature in selected_features:
        churned = df[(df['FEATURE'] == feature) & df['MSISDN_ENCR_INT'].isin(result_df[result_df['Predicted_Churn'] == 1]['MSISDN_ENCR_INT'])]
        not_churned = df[(df['FEATURE'] == feature) & df['MSISDN_ENCR_INT'].isin(result_df[result_df['Predicted_Churn'] == 0]['MSISDN_ENCR_INT'])]

        day_cols = [col for col in churned.columns if col.startswith("DAY")]
        if not churned.empty and not not_churned.empty:
            avg_churned = churned[day_cols].mean()
            avg_not_churned = not_churned[day_cols].mean()
            fig, ax = plt.subplots()
            ax.plot(avg_churned.index, avg_churned.values, label="Churned", marker='o')
            ax.plot(avg_not_churned.index, avg_not_churned.values, label="Not Churned", marker='o')
            ax.set_title(f"{feature}: Average Values Over 30 Days")
            ax.set_xlabel("Day")
            ax.set_ylabel("Average Value")
            ax.set_xticks([0, 5, 10, 15, 20, 25, 30])
            ax.set_xticklabels([f"Day {i}" for i in range(0, 31, 5)])
            ax.legend()
            _, fwa_col2, _ = st.columns([1, 3, 1])
            with fwa_col2:
                st.pyplot(fig, use_container_width=True)
        else:
            st.info(f"Not enough data to plot {feature}.")


def render():
    """
    The main function to render the Streamlit app.
    This function sets up the page configuration, title, and handles the different sections of the app for batch prediction page.
    """

    st.header("Batch Churn Prediction")
    # Browse for file and load data from CSV
    userid_df = section_load_data_element()
    prediction_df = extract_prediction_data(userid_df, st.session_state.uploaded_db_df)

    # User selection for model and threshold
    model_file, threshold = section_select_model_and_threshold()

    # Run prediction
    prediction_disabled = "userid_df" not in st.session_state
    if st.button("Run Prediction", disabled=prediction_disabled):
        st.session_state.threshold = threshold

        # Run the prediction and store results in session state
        result_df = section_run_prediction(prediction_df, model_file, threshold)
        st.session_state.result_df = result_df

    # Display results
    if "result_df" in st.session_state:
        result_df = st.session_state.result_df

        st.subheader("Prediction Results")
        _, pr_col2, _ = st.columns([1, 5, 1]) # Center the results in the middle column
        with pr_col2:
            st.dataframe(result_df.head(), use_container_width=True)
        csv = result_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Predictions", csv, "churn_predictions.csv", "text/csv")
        section_plot_churn_distribution(result_df)

        st.subheader("Feature Analysis by Churn")
        section_feature_wise_analysis(prediction_df, result_df)


if __name__ == "__main__":
    # Keep this for streamlit to run the app automatically using native multi-page support
    render()