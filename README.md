# Churn Prediction App - Streamlit

## SETUP ENVIRONMENT
* Download python 3.8 from python official website: `https://www.python.org/downloads/`
* Install python to preferred directory: `C:\Users\Pawan\environments\pythons\python308`
* - No need to install documentation or test suits or add to path
* On command prompt, create a venv using downloaded python version: `"C:\Users\Pawan\environments\pythons\python308\python.exe" -m venv "C:\Users\Pawan\environments\venvs\supipi_py308"`
* Activate the environment: `C:\Users\Pawan\environments\venvs\supipi_py308\Scripts\activate`
* Go to the churn project directory: `cd C:\backup\personal\Supipi\churn`
* Install libraries: `pip install -r requirements.txt`  


## REQUISITES
* Have the model file saved in the `model` directory. It should be a _.pkl_ file
* Have the customer database file saved in the `data` directory. It should be a _.csv_ file  


## RUN STREAMLIT APP
* Run the following command on the terminal: `streamlit run Home.py`
* Open the local URL or Network URL in your default web browser  

### FEATURES
* The app has 5 pages accessible through the navigation panel on the left
* Home page has high level information on the app and which pages to try
* Batch Predict page can be used to get the churn prediction result for a set of User IDs uploaded through a _.csv_ file
* Single Predict page can be used to get the churn prediction result for a single User ID selected from an uploaded _.csv_ file or by entering the Usser ID manually. It has SHAP feature importance visualizations and feature trends for the selected user.
* Database page shows the default database dataframe read by the app. If required, the database file could be updated by uploading a new file
* About page shows information regarding the project  


## UPDATING CODE
* While running the streamlit app, you can change the script which will update the app in realtime based on update logic in the script
