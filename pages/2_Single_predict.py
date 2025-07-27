import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from utils.db_utils import extract_prediction_data
from utils.model_prediction import ChurnPrediction
from utils.data_process import create_prediction_data
from utils.gui_utils import section_load_data_element, section_select_model_and_threshold, section_run_prediction, process_data

rng = np.random.default_rng(42)  # For reproducibility


def get_background_data_for_SHAP(num_users=10):
    """
    Get background data for SHAP calculations. Background data is used to simulate what the prediction
    would be if a feature was missing. It is required for DeepExplainer to compute SHAP values.
    The processing method was taken from the training ipynb script.
    
    :param num_users: Number of users to sample from the database.
    :return: Numpy array containing background data for SHAP.
    """

    db_df = st.session_state.uploaded_db_df
    unique_users = db_df['MSISDN_ENCR_INT'].dropna().unique()
    sampled_users = rng.choice(unique_users, size=min(num_users, len(unique_users)), replace=False)
    sampled_df = db_df[db_df['MSISDN_ENCR_INT'].isin(sampled_users)]
    background_processed = create_prediction_data(sampled_df)
    X_background = np.concatenate([x.reshape((x.shape[0], -1)) for x in background_processed], axis=1)

    return X_background


def create_shap_explainer(num_users=10):
    """
    Create a SHAP explainer using the background data for SHAP calculations.
    Future features: Create an input to get the num_users from the user interface to allow SHAP calculation
    robustness. Needs more data.
    
    :param num_users: Number of users to sample for background data.
    :return: SHAP explainer object.
    """

    def model_predict(x_flat):
        xs = np.split(x_flat, 6, axis=1)
        xs = [x.reshape((x.shape[0], 30, 1)) for x in xs]
        return model.predict(xs).ravel()
    
    background_data = get_background_data_for_SHAP(num_users)
    predictor = st.session_state.predictor
    model = predictor.get_model()
    explainer = shap.KernelExplainer(model_predict, background_data)
    return explainer


def plot_horizontal_bar_chart(shap_values, feature_names):
    """
    Plot a horizontal bar chart for SHAP values.
    
    :param shap_values: SHAP values to plot.
    :param feature_names: Names of the features corresponding to the SHAP values.
    """

    _, sfp_col2, _ = st.columns([1, 3, 1])
    with sfp_col2:
        fig, ax = plt.subplots()
        ax.barh(feature_names, shap_values, color='skyblue')
        ax.set_xlabel("SHAP Value")
        ax.set_title("SHAP Feature Contributions")
        ax.invert_yaxis()  # Invert y-axis to have the highest SHAP value on top
        st.pyplot(fig)


@st.cache_data
def section_shap_explanations(single_user_prediction_df):
    """
    Section defining the UI elements for SHAP feature calculation and visualization.
    Future features: add feature to select the number of user for background data and also how many features to show.

    :param single_user_prediction_df: DataFrame containing the single user data for prediction.
    """

    explainer = create_shap_explainer(num_users=10)
    input_data_for_user = process_data(single_user_prediction_df)
    x_test_flat = np.concatenate([x.reshape((x.shape[0], -1)) for x in input_data_for_user], axis=1)
    shap_values = explainer.shap_values(x_test_flat)

    # Calculate the top 5 features with highest SHAP values
    shap_values_reshaped = shap_values.reshape((6, 30))
    flat_indices = np.argpartition(np.abs(shap_values_reshaped).ravel(), -5)[-5:]
    top_indices = flat_indices[np.argsort(np.abs(shap_values_reshaped).ravel()[flat_indices])[::-1]]
    row_indices, column_indices = np.unravel_index(top_indices, shap_values_reshaped.shape)
    top_positions = list(zip(row_indices, column_indices))

    all_feature_names = single_user_prediction_df['FEATURE'].dropna().unique()
    top_features = [f"{all_feature_names[row]} of Day {col + 1}" for (row, col) in top_positions]
    top_values = [shap_values_reshaped[row, col] for (row, col) in top_positions]
    plot_horizontal_bar_chart(top_values, top_features)
    

def show_feature_trends(user_raw_df):
    """
    Function to show feature trends for the user for the 30 days for each feature.

    :param user_raw_df: DataFrame containing the data for the user.
    """

    features = user_raw_df['FEATURE'].unique()
    _, ft_col2, _ = st.columns([1, 3, 1])
    for feature in features:
        feature_data = user_raw_df[user_raw_df['FEATURE'] == feature]
        if feature_data.empty:
            continue
        day_cols = [col for col in feature_data.columns if col.startswith('DAY')]
        values = feature_data[day_cols].values.flatten()
        with ft_col2:
            fig, ax = plt.subplots()
            ax.plot(range(1, 31), values, marker='o')
            ax.set_title(f"{feature} Trend Over Last 30 Days")
            ax.set_xlabel("Day")
            ax.set_ylabel("Value")
            st.pyplot(fig)


def render():
    """
    The main function to render the Streamlit app.
    This function sets up the page configuration, title, and markdown content for single prediction page.
    """
    
    st.header("Single Customer Prediction")

    input_method = st.radio("Select UserID Input Method", ["Select from uploaded CSV", "Manual entry"])
    # selected_userid = None
    if input_method == "Select from uploaded CSV":
        userid_df = section_load_data_element()
        if not userid_df.empty:
            user_ids = userid_df[st.session_state.id_column_name].unique()
            selected_userid = st.selectbox("Select UserID", user_ids)
        else:
            selected_userid = None
    elif input_method == "Manual entry":
        selected_userid = st.text_input("Enter UserID")
        if selected_userid != '':
            selected_userid = np.float64(selected_userid)

    model_file, threshold = section_select_model_and_threshold()

    if selected_userid:
        single_user_prediction_df = extract_prediction_data(
            pd.DataFrame({st.session_state.id_column_name: [selected_userid]}), 
            st.session_state.uploaded_db_df
            )
        if single_user_prediction_df.empty:
            st.warning(f"No data found for UserID: {selected_userid}")
            return
        
        # Run prediction
        if st.button("Run Prediction"):
            st.session_state.threshold = threshold

            # Run the prediction and store results in session state
            result_df = section_run_prediction(single_user_prediction_df, model_file, threshold)
            st.session_state.result_df = result_df

            if 'result_df' in st.session_state:
                churn_probability = st.session_state.result_df['Churn_Probability'].values[0]
                churn_label = st.session_state.result_df['Predicted_Churn'].values[0]

                st.subheader("Prediction Results")
                st.write(f"Churn Probability: {churn_probability:.2f}")
                st.write(f"Predicted Churn: {'Yes' if churn_label == 1 else 'No'}")

                st.subheader("SHAP Feature Contributions")
                section_shap_explanations(single_user_prediction_df)

                st.subheader("30-Day Feature Trends")
                show_feature_trends(single_user_prediction_df)
            

if __name__ == "__main__":
    # Keep this for streamlit to run the app automatically using native multi-page support
    render()