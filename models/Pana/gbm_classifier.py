"""
gbm_classifier.py
-----------------

This module defines a GBM model for classifying emotions based on features extracted from Mel-Spectrogram images.

Functions:
- train_gbm_model: Trains a GBM model on the extracted features.
- evaluate_model: Evaluates the trained GBM model.
"""
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def train_gbm_model(features, labels, test_size=0.2, random_state=42):
    """
    Trains a GBM model using XGBoost on the provided features and labels.

    Parameters:
    - features: np.array, the feature set extracted from the images.
    - labels: np.array, the labels corresponding to each feature set.
    - test_size: float, the proportion of the dataset to include in the test split.
    - random_state: int, the seed used by the random number generator.

    Returns:
    - model: The trained XGBoost model.
    - eval_result: The evaluation result on the test set.
    """
    # Splitting the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size,
                                                        random_state=random_state)

    # Initialize an XGBoost classifier
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions on the test set
    predictions = model.predict(X_test)

    # Evaluate the model
    eval_result = classification_report(y_test, predictions)
    print("Classification Report:\n", eval_result)

    return model, eval_result
