from sklearn.mixture import GaussianMixture


def create_gmm_models(n_components=16, n_classes=7):
    gmm_models = {class_index: GaussianMixture(n_components=n_components, covariance_type='full', random_state=42) for
                  class_index in range(n_classes)}
    return gmm_models
