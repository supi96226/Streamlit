import streamlit as st
import pandas as pd
from utils.db_utils import load_csv
from utils.data_process import create_prediction_data
from utils.model_prediction import ChurnPrediction
import os

# Constants. Use these to define default model directory, model name, and threshold.
DEFAULT_MODEL_DIR = 'model'
DEFAULT_MODEL_NAME = 'best_model_final.pkl'
DEFAULT_THRESHOLD = 0.50


def get_model_file_dropdown():
    """
    Returns a list of model files available in the default model directory.
    This function is used to populate a dropdown for model selection in the Streamlit app.

    :return: List of model file names.
    """

    model_files = [f for f in os.listdir(DEFAULT_MODEL_DIR) if f.endswith(".pkl")]
    return model_files


def section_select_model_and_threshold():
    """
    Section defining the UI elements for model selection and threshold input.
    This function allows users to select a model from a dropdown and set a churn threshold.

    :return: Selected model file name and threshold value.
    """

    col1, col2 = st.columns([2, 1])
    with col1:
        model_files = get_model_file_dropdown()
        model_file = st.selectbox("Select Model", model_files, index=model_files.index(DEFAULT_MODEL_NAME))
    with col2:
        if "threshold" not in st.session_state:             # Initialize threshold in session state if not present
            st.session_state.threshold = DEFAULT_THRESHOLD
        threshold = st.number_input("Churn Threshold", value=st.session_state.threshold, min_value=0.0, max_value=1.0, step=0.01)   
        # st.session_state.threshold = threshold            # This line was put inside run prediction because it caused lag when changing threshold
    return model_file, threshold


def section_load_data_element():
    """
    Section defining the UI elements in the load data section.
    This function allows users to upload a UserID CSV file that needs the churn prediction.
    It also handles the case where a new file is uploaded, clearing previous data if necessary.

    :return: DataFrame containing the uploaded data.
    """

    # Upload CSV file
    uploaded_user_file = st.file_uploader("Upload UserID CSV", type=["csv"])
    uuf_col1, uuf_col2 = st.columns([1, 2])

    # ### Use this part if you wan to cache the uploaded user files.
    # ### However this will cause the app to not reload the file when just the data is changed, not the file name.
    # if uploaded_user_file:
    #     if "last_uploaded_user_file" not in st.session_state or st.session_state.last_uploaded_user_file != uploaded_user_file.name:
    #         st.session_state.pop("userid_df", None)   # Clear previous data if a new file is uploaded
    #         st.session_state.last_uploaded_user_file = uploaded_user_file.name
    if uploaded_user_file is not None:
        st.session_state.pop("userid_df", None)  # Clear previous data if a new file is uploaded
    # ### End of caching part based on filename

    if "userid_df" in st.session_state:               # Reuse previously uploaded data if available
        userid_df = st.session_state.userid_df
        with uuf_col1:
            st.dataframe(userid_df.head())
        with uuf_col2:
            st.markdown(f"""
                **UserID CSV File Details:**
                - Column '{st.session_state.id_column_name}' is extracted for User IDs
                - Number of UsersIDs:    **{userid_df.shape[0]}**
                - Please click 'Run Prediction' to proceed with the prediction.
            """)
    else:                                               # Load new data if no previous upload exists
        if uploaded_user_file is not None:
            userid_df = load_csv(uploaded_user_file)
            st.session_state.userid_df = userid_df
            with uuf_col1:
                st.dataframe(userid_df.head())
            with uuf_col2:
                id_column = userid_df.columns[0] if not userid_df.empty else "MSISDN_ENCR_INT"
                st.session_state.id_column_name = id_column  # Store the ID column name in session state
                st.markdown(f"""
                    **UserID CSV File Details:**
                    - Column '{st.session_state.id_column_name}' is extracted for User IDs
                    - Number of UsersIDs:    **{userid_df.shape[0]}**
                    - Please click 'Run Prediction' to proceed with the prediction.
                """)
        else:
            userid_df = pd.DataFrame()                     # Empty DataFrame if no file uploaded
    return userid_df


@st.cache_data
def process_data(df):
    """
    Wrapper to the implemented function to preprocess the DataFrame for churn prediction.

    :param df: DataFrame containing raw data.
    :return: Preprocessed DataFrame ready for prediction.
    """

    return create_prediction_data(df)


@st.cache_data
def get_predictions(model_file, input_data):
    """
    Function to get predictions from the ChurnPrediction model.
    This function caches the predictions to avoid reloading the model and running prediction multiple times.

    :param model_file: Name of the model file to load.
    :param input_data: Preprocessed input data for prediction.
    :return: Predictions from the model.
    """

    predictor = ChurnPrediction(os.path.join(DEFAULT_MODEL_DIR, model_file))
    st.session_state.predictor = predictor  # Store the predictor in session state for later use
    predictions = predictor.predict_batch(input_data)
    return predictions


def section_run_prediction(raw_df, model_file, threshold):
    """
    Section defining the UI elements for running the churn prediction
    This function processes the uploaded data, runs the prediction using the selected model, and returns the results
    Since predictions are cached, the churn labels can be computed without reloading the model each time.

    :param raw_df: DataFrame containing the uploaded data.
    :param model_file: Name of the model file to use for prediction.
    :param threshold: Threshold value for churn classification.
    :return: DataFrame containing the churn prediction results.
    """

    input_data = process_data(raw_df)
    predictions = get_predictions(model_file, input_data)
    churn_prob = predictions.flatten()
    churn_label = (churn_prob >= threshold).astype(int)
    msisdn_list = raw_df["MSISDN_ENCR_INT"].dropna().unique()
    result_df = pd.DataFrame({
        "MSISDN_ENCR_INT": msisdn_list,
        "Churn_Probability": churn_prob,
        "Predicted_Churn": churn_label
    })
    return result_df