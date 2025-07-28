import pandas as pd
import numpy as np


def read_data_csv(file_path):
    """
    Reads a CSV file and returns a DataFrame.

    :param file_path: Path to the CSV file.
    :return: DataFrame containing the data.
    """

    return pd.read_csv(file_path)


def preprocess_data(df):
    """
    Preprocesses the DataFrame for churn prediction. Preprocessing includes, removing unnecessary columns,
    handling missing values, and reshaping the data.

    :param df: DataFrame containing raw data.
    :return: Preprocessed DataFrame.
    """

    day_cols = [f'DAY{i}' for i in range(1, 31)]
    df[day_cols] = df[day_cols].replace('\\N', np.nan)
    df[day_cols] = df[day_cols].apply(pd.to_numeric, errors='coerce')

    def treat_missing_values(group):
        feature = group['FEATURE'].iloc[0]
        if feature in ['TOTAL_REVENUE_WT', 'TOTAL_VOICE_USAGE', 'MBB_TOTAL_USAGE', 'PAYG_MB_RATIO']:
            # Fill NaNs with 0.0
            group[day_cols] = group[day_cols].fillna(0.0)
        elif feature in ['WALLET_BALANCE', 'REMAINING_DATA_BAL_MB']:
            # Interpolate across the 30 days (row-wise), fill any remaining NaNs
            group[day_cols] = group[day_cols].interpolate(axis=1, limit_direction='both').fillna(0.0)
        return group

    df = df.groupby('FEATURE', group_keys=False).apply(treat_missing_values)

    return df


def transform_data(X):
    """
    Transforms the input data for prediction by standardizing and reshaping it to fit the model's expected 
    input shape. 
    Add standard scaling by reading the scaler calculated during training. For now using 1.
    # scaler = # Load the scaler from a file or define it here
    # X_scaled = scaler.transform(X)
    
    :param X: Input data to be transformed
    :return: Transformed data.
    """

    X_scaled = X * 1  # Placeholder for actual scaling
    X_reshaped = np.reshape(X_scaled.values, (X_scaled.shape[0], X_scaled.shape[1], 1))
    return X_reshaped


def create_prediction_data(file_path_or_df):
    """
    Ensemble of preprocessing and restructuring functions to read data from a CSV file for prediction. 
    This is the top level function that combines all steps.

    :param file_path: Path to the CSV file.
    :return: Preprocessed DataFrame ready for prediction.
    """

    if isinstance(file_path_or_df, pd.DataFrame):
        raw_df = file_path_or_df
    else:
        if not isinstance(file_path_or_df, str):
            raise ValueError("Input must be a file path or a DataFrame.")
        else:
            raw_df = read_data_csv(file_path_or_df)

    preprocessed_df = preprocess_data(raw_df)

    feature_map = {
        "TOTAL_REVENUE_WT": None,
        "TOTAL_VOICE_USAGE": None,
        "WALLET_BALANCE": None,
        "MBB_TOTAL_USAGE": None,
        "PAYG_MB_RATIO": None,
        "REMAINING_DATA_BAL_MB": None
    }

    for feature in feature_map.keys():
        feature_map[feature] = preprocessed_df[preprocessed_df['FEATURE'] == feature].drop(columns=['MSISDN_ENCR_INT', 'FEATURE'])
    
    resampled_sets = []
    for feature, data in feature_map.items():
        if data is not None and not data.empty:
            try:
                X_res = data.drop(columns=['CHURN_TAG'])
                y_res = data['CHURN_TAG']
                resampled_sets.append((X_res, y_res))
            except:
                print('Error in extracting data from Dataframe')
        else:
            print(f"[WARNING] Feature '{feature}' is missing or empty. Skipping.")

    processed_sets = [transform_data(X) for X, _ in resampled_sets]

    x1, x2, x3, x4, x5, x6 = processed_sets
    prediction_data = [x1, x2, x3, x4, x5, x6]
    return prediction_data


if __name__ == "__main__":
    prediction_data = create_prediction_data('data/Input data DL.csv')
    print(np.asarray(prediction_data).shape)

