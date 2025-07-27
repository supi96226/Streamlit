from utils.model_load import load_model_from_pickle
from utils.data_process import create_prediction_data
import numpy as np


class ChurnPrediction:
    """
    Class for predicting customer churn using a trained deep learning model.
    """
    
    def __init__(self, model_path):
        """
        Initializes the ChurnPrediction class with a trained model

        :param model: A trained deep learning model for churn prediction.
        """

        # model_path = 'data/best_model_final.pkl'
        self.model = load_model_from_pickle(model_path)

    
    def get_model(self):
        """
        Returns the loaded model.

        :return: The trained model.
        """
        return self.model


    def predict_batch(self, input_data):
        """
        Predicts churn based on input data.

        :param input_data: A dataframe containing features for prediction.
        :return: The predicted churn probability.
        """

        # Make prediction
        prediction = self.model.predict(input_data) # Probability of churn
        return prediction
    

    def predict_single(self, input_data):
        """
        Predicts churn for a single instance.

        :param input_data: A dataframe containing features for prediction.
        :return: The predicted churn probability.
        """

        # Make prediction
        replicated_input = list(np.expand_dims(input_data, axis=1))
        prediction = self.model.predict(replicated_input)
        return prediction


if __name__ == "__main__":
    model_path = 'data/best_model_final.pkl'
    churn_predictor = ChurnPrediction(model_path)
    input_data = create_prediction_data('data/Input data DL.csv')
    
    # Predict for a batch of data
    predictions = churn_predictor.predict_batch(input_data)
    print("Batch Predictions:", predictions)

    # Predict for a single instance
    single_input = [x[0] for x in input_data]
    single_prediction = churn_predictor.predict_single(single_input)
    print("Single Prediction:", single_prediction)