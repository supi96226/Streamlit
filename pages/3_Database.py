import streamlit as st
import pandas as pd
from utils.db_utils import load_csv


def render():
    """
    The main function to render the Streamlit app.
    This function sets up the page configuration, title, and markdown content for about page.
    """
    
    st.header("Update Database")
    st.markdown("""
        **Database Page** allows you to update the customer database with new data.
        In case the default database is needs to be changed, you can upload a new CSV file to replace the existing one.
    """)

    # If a new database is needed, upload a new CSV file
    uploaded_db_file = st.file_uploader("Upload Customer Database CSV file", type=["csv"])
    if uploaded_db_file:
        if "last_uploaded_db_file" not in st.session_state or st.session_state.last_uploaded_db_file != uploaded_db_file.name:
            st.session_state.pop("uploaded_db_df", None)   # Clear previous data if a new file is uploaded
            st.session_state.last_uploaded_db_file = uploaded_db_file.name
    if "uploaded_db_df" in st.session_state:               # Reuse previously uploaded data if available
        raw_db_df = st.session_state.uploaded_db_df
        st.dataframe(raw_db_df.head(10))
        st.write(f"Database shape: {raw_db_df.shape[0]} rows, {raw_db_df.shape[1]} columns")
    else:                                               # Load new data if no previous upload exists
        if uploaded_db_file is not None:
            raw_db_df = load_csv(uploaded_db_file)
            st.session_state.uploaded_db_df = raw_db_df
            st.dataframe(raw_db_df.head(10))
            st.write(f"Database shape: {raw_db_df.shape[0]} rows, {raw_db_df.shape[1]} columns")
        else:
            raw_db_df = pd.DataFrame()                     # Empty DataFrame if no file uploaded


if __name__ == "__main__":
    # Keep this for streamlit to run the app automatically using native multi-page support
    render()