import pickle
import logging
from model_script.svm import MultiClassSVM_Vectorized
from model_script.naive_bayes import ComplementNaiveBayes
import logging
from flask import has_request_context, g

logger = logging.getLogger(__name__)

class RequestIDFormatter(logging.Formatter):
    def format(self, record):
        if has_request_context():
            record.request_id = getattr(g, 'request_id', 'no-request-id')
        else:
            record.request_id = 'startup'
        return super().format(record)

def setup(path):
    try:
        loading_model_conv, is_model_loaded = load_file(path)
        logger.debug(f"loading model and weights for {loading_model_conv['metadata']['svm_versi']} and {loading_model_conv['metadata']['cnb_versi']}")

        logger.info("Initializing models")
        logger.info("Initializing TF-IDF Vectorizer")
        
        tf_idf = loading_model_conv["tfidf"]

        svm = MultiClassSVM_Vectorized(W=loading_model_conv['svm_W'], b=loading_model_conv['svm_b'], classes=loading_model_conv['svm_classes'])
        naive_bayes = ComplementNaiveBayes(feature_prob_log=loading_model_conv["cnv_prob_log"], classes=loading_model_conv["cnb_classes"])
        return loading_model_conv, is_model_loaded, tf_idf, svm, naive_bayes
    except Exception as e:
        # logger.exception(f"Error during initialization: {str(e)}")
        logger.exception("Error during initialization")
        raise

def load_file(file_path):
    logger.debug(f"Loading file {file_path}")
    with open(file_path, 'rb') as f:
        return pickle.load(f), True

def tfidf_transform(text : str, tf_idf):
    try:
        return tf_idf.transform([text]).toarray()
    except Exception as e:
        logger.exception("Error during TF-IDF transformation")
        raise

class ValidationError(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message

def validate_text_request(data):
    data = data
    logger.debug(f"Validating request data: {data}")
    if data is None:
        raise ValidationError("Invalid request: JSON body is missing")
    elif 'text' not in data:
        raise ValidationError("Invalid request: 'text' field is missing")
    elif not isinstance(data['text'], str):
        raise ValidationError("Invalid requesst: 'text' field must be a string")
    elif not data['text'].strip():
        raise ValidationError("Invalid request: 'text' field must not be an empty string")

    return data['text']