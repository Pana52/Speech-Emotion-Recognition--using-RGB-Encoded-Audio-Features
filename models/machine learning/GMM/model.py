from sklearn.mixture import GaussianMixture


def create_gmm_models(n_components=16, n_classes=6):
    """
    Creates a dictionary of Gaussian Mixture Models, one for each class.

    Parameters:
    - n_components: The number of mixture components for each GMM.
    - n_classes: The number of classes (emotions) in the dataset.

    Returns:
    A dictionary where keys are class indices (0 to n_classes-1) and values are the GMM instances.
    """
    gmm_models = {class_index: GaussianMixture(n_components=n_components, covariance_type='full', random_state=42) for
                  class_index in range(n_classes)}
    return gmm_models
