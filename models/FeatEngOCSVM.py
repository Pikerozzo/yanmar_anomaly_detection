from sklearn.svm import OneClassSVM
import numpy as np

class FeatEngOCSVM():
    def __init__(self, kernel='rbf', nu=0.01, gamma='scale'):
        self.model = OneClassSVM(kernel=kernel, nu=nu, gamma=gamma)
        self.fitted = False

        self.gamma = self.model.gamma if self.model.gamma in ['scale', 'auto'] else float(self.model.gamma)
        self.nu = self.model.nu
        self.kernel = self.model.kernel
        self.model_name = self.__class__.__name__
        self.fitted = False
        self.__supports = None  # to store support vectors after fitting

    def fit(self, X):
        self.model.fit(X)
        self.fitted = True
        self.__supports = self.model.support_vectors_

    @property
    def supports(self):
        if not self.fitted:
            raise RuntimeError("The model must be fitted before accessing support vectors.")
        return self.__supports

    def decision_function(self, X):
        # X: precomputed kernel matrix between test samples and support vectors
        scores = self.model.decision_function(X)
        return scores

    def predict(self, X):
        if not self.fitted:
            raise RuntimeError("The model must be fitted before prediction.")
        pred_scores = self.model.predict(X)
        pred_labels = self._extract_labels(pred_scores)
        return pred_labels

    def score_samples(self, X):
        return self.model.score_samples(X)

    def _extract_labels(self, df_results):
        return np.where(df_results > 0, 0, 1)