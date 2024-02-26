# model.py
from hmmlearn import hmm


def create_hmm_model(n_components=8):
    """
    Creates and returns a Hidden Markov Model.

    :param n_components: The number of states in the model.
    :return: The HMM model.
    """
    # Assuming a Gaussian HMM for simplicity; adjust as needed
    model = hmm.GaussianHMM(n_components=n_components, covariance_type="diag", n_iter=1000)
    return model
