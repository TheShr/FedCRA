from sklearn.preprocessing import StandardScaler
import numpy as np
from numpy.typing import NDArray
from log_config import base_logger
logger = base_logger(__name__)
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import pandas as pd
from pandera.typing import DataFrame

from typing import List

class DataPreprocessing:
    @staticmethod
    def min_max_normalization(data: NDArray) -> NDArray:
        """
        Normalize data using min-max normalization.
        :param data: np.array input data
        :return: np.array Normalized data
        """
        min_val = np.min(data)
        max_val = np.max(data)
        if min_val == max_val:
            # Handle the case when min_val and max_val are the same
            normalized_data = np.zeros_like(data)
        else:
            normalized_data = (data - min_val) / (max_val - min_val)
        return normalized_data

    @staticmethod
    def standard_scaling(data: NDArray) -> NDArray:
        """
        Scale data using standard scalar, also called z-score normalization.
        :param data: np.array input data
        :return:  np.array Standard scaled
        """
        logger.info('Data Scaled using Standard Scalar')
        standard_scalar = StandardScaler()
        scaled_data = standard_scalar.fit_transform(data)
        return scaled_data


def encoding(data_frame: DataFrame, features: List, encoding_type: str = 'onehot_encoding'):
    """
    Encodes the specified features in the DataFrame using one-hot or label encoding and returns the updated DataFrame.
    Args:
        data_frame : The DataFrame containing the data to encode.
        features : The columns to encode.
        encoding_type: The type of encoding ('onehot_ecoding' or 'label_encoding').
    Returns:
        pd.DataFrame: DataFrame with the encoded features integrated.
    """
    df = data_frame.copy()
    if encoding_type == 'onehot_encoding':
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoded_data = encoder.fit_transform(df[features])
        encoded_columns = encoder.get_feature_names_out(features)
        encoded_df = pd.DataFrame(encoded_data, columns=encoded_columns, index=df.index)
        df = df.drop(columns=features).join(encoded_df)
    elif encoding_type == 'label_encoding':
        if isinstance(features, list):
            for feature in features:
                encoder = LabelEncoder()
                df[feature] = encoder.fit_transform(df[feature])
        else:
            encoder = LabelEncoder()
            df[features] = encoder.fit_transform(df[features])
    else:
        raise ValueError("Invalid encoding_type. Choose 'onehot' or 'label'.")
    return df