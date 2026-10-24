import numpy as np
np.random.seed(42)
import logging

logger = logging.getLogger(__name__)


class MultiClassSVM_Vectorized:
    def __init__(self, learning_rate=None, lambda_parameter=0.01, n_iters=None, W=None, b=None, classes=None):
        logger.debug("Initialling SVM")
        self.learning_rate = learning_rate
        self.lambda_parameter = lambda_parameter
        self.n_iters = n_iters
        self.W = W
        self.b = b
        self.classes_ = classes


    def predict(self, X):
        logger.debug("Predicting using SVM")
        scores = np.dot(X, self.W.T) - self.b
        best_indices = np.argmax(scores, axis=1)
        return self.classes_[best_indices]