from sklearn.svm import SVC


def create_model():
    """
    Creates and returns a Support Vector Machine (SVM) model
    with specified parameters.
    """
    model = SVC(kernel='linear', probability=True, C=1.0)  # C is the regularization parameter

    return model
