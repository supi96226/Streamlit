import streamlit as st
from utils.data_process import read_data_csv

# Constants. Use these to define default customer database file.
DATABASE_PATH = "data/Customer_Database.csv"


@st.cache_data
def load_csv(uploaded_file):
    """
    Wrapper to the implemented function to load a CSV file into a DataFrame.

    :param uploaded_file: Uploaded CSV file.
    :return: DataFrame containing the data from the CSV file.
    """

    return read_data_csv(uploaded_file)


@st.cache_data
def extract_prediction_data(userid_df, db_df):
    """
    Extracts the prediction data from the user and database DataFrames.

    :param userid_df: DataFrame containing user data.
    :param db_df: DataFrame containing database data.
    :return: List of processed data ready for prediction.
    """

    # Extract users given in userid_df from the database
    if userid_df.empty:
        return []
    filtered_df = db_df[db_df['MSISDN_ENCR_INT'].isin(userid_df[st.session_state.id_column_name])]
    if filtered_df.empty:
        st.warning("No matching records found in the database for the selected user IDs.")
        return []
    return filtered_df


def init_db():
    """
    Initialize the database by loading the default customer database.
    This function is called in Home page when the app starts to ensure the database is ready for use.
    """
    
    db_df = load_csv(DATABASE_PATH)
    st.session_state.uploaded_db_df = db_df  # Store the initial data in session state
    st.session_state.last_uploaded_db_file = None  # Track the last uploaded file name


if __name__ == "__main__":
    # This is just a placeholder to allow the module to be run directly for testing purposes
    init_db()
