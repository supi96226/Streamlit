import streamlit as st


def render():
    """
    The main function to render the Streamlit app.
    This function sets up the page configuration, title, and markdown content for about page.
    """
    
    st.header("About this App")
    st.markdown("""
        **Customer Churn Prediction App** built using Streamlit.

        - Developed by Supipi Munasinghe in partial fulfillment of the requirements for the Master of Science degree final year project
        - Predicts churn using a Composite deep learning model combining BiLSTM and CNN layers with transformer encoders trained on 30-day behavioral usage data
        - Utilizes custom focal loss to handle class imbalance
        
        Libraries used: Streamlit, TensorFlow, Pandas, NumPy, Matplotlib, Seaborn
    """)


if __name__ == "__main__":
    # Keep this for streamlit to run the app automatically using native multi-page support
    render()