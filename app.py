import streamlit as st
from utils.db_utils import init_db

st.set_page_config(page_title="Customer Churn Prediction App", layout="wide")


def initialize_features():
    """
    Initialize the features required for the app.
    This function is called in Home page when the app starts to ensure all necessary features are ready for use.
    """
    
    # Initialize the database
    init_db()


def render_app():
    """
    The main function to render the Streamlit app - Home page.
    This function sets up the page configuration, title, and markdown content for the main app page
    """

    st.markdown(
        """
        <style>
            .stApp {
                background-color: #fdf6e3;
            }
        </style>
        """,
        unsafe_allow_html=True
    )


    # st.set_page_config(page_title="Customer Churn Prediction App", layout="wide")
    st.title("Customer Churn Prediction App")

    st.markdown("""
        ## Welcome

        This app predicts **customer churn** using a deep learning model trained on behavioral features.

        ### How to Use:
        - Open the **navigation sidebar** to access different sections
        - Go to **Batch Prediction** to upload a full dataset and analyze churn risk
        - Go to **Single Prediction** to choose a user from the data and analyze churn prediction with SHAP explanations
        - View the **About** section for project background

        The model uses a Deep Learning Composite moddel combining BiLSTM and CNN layers with Transformer Encoders to predict the churn probabilities with visual insights.
    """)

    # Initialize features required for the app
    initialize_features()


if __name__ == "__main__":
    render_app()
    