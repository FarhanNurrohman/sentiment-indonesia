import numpy as np
import logging

logger = logging.getLogger(__name__)


class ComplementNaiveBayes():
    def __init__(self, feature_prob_log=None, classes=None):
        logger.debug("Initialling Naive Bayes")
        self._feature_prob_log = feature_prob_log
        self._classes = classes

    def predict(self, X):
        logger.debug("Predicting using Naive Bayes")
        y_pred = np.dot(X, self._feature_prob_log.T)
        return self._classes[np.argmin(y_pred, axis=1)]
        