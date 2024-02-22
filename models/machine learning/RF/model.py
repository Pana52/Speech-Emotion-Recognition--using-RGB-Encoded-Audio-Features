from sklearn.ensemble import RandomForestClassifier


def create_model(n_estimators=100, random_state=42):
    """
    Create a Random Forest model with specified parameters.

    Parameters:
    - n_estimators: The number of trees in the forest.
    - random_state: Controls both the randomness of the bootstrapping of the samples used when building trees (if bootstrap=True) and the sampling of the features to consider when looking for the best split at each node (if max_features < n_features).

    Returns:
    - A RandomForestClassifier model instance.
    """
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    return model
