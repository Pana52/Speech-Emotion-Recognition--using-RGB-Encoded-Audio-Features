from sklearn.ensemble import RandomForestClassifier


def create_model(n_estimators=100, random_state=42):

    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    return model
