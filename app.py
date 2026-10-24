from helper import validate_text_request, tfidf_transform, setup, ValidationError, RequestIDFormatter
from flask import Flask, request, jsonify, g
import logging
import os
import uuid
from waitress import serve

level = os.getenv("LOG_LEVEL", "INFO").upper()

tokenizer_path = os.getenv('TOKENIZER_PATH', 'model_weights/indobert_tokenizer/')
conv_model_path = os.getenv('CONV_MODEL_PATH', 'model_weights/conv_model_bundle.pkl')
indobert_model_path = os.getenv('INDOBERT_MODEL_PATH', 'model_weights/model_weights_np.pkl')

vocab_size = int(os.getenv('VOCAB_SIZE', '50000'))
d_model = int(os.getenv('D_MODEL', '768'))
d_ff = int(os.getenv('D_FF', '3072'))

root = logging.getLogger()
root.setLevel(level)

handler = logging.StreamHandler()
handler.setFormatter(RequestIDFormatter("%(asctime)s - %(levelname)s - [%(request_id)s] - %(message)s"))

root.addHandler(handler)
logger = logging.getLogger(__name__)

app = Flask(__name__)

loading_model_conv, is_model_loaded, tf_idf, svm, naive_bayes = setup(conv_model_path)

@app.before_request
def before_request():
    incoming = request.headers.get('X-Request-ID')
    g.request_id = incoming if incoming else str(uuid.uuid4())[:8]
    logger.info(f"Request start: {request.method} {request.path}")

@app.after_request
def after_request(response):
    response.headers['X-Request-ID'] = getattr(g, 'request_id', 'unknown')
    return response

@app.route('/', methods=['GET'])
def home():
    return "Welcome to the Sentiment Analysis API!"

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "ok", 
        "model_loaded": is_model_loaded,
        "svm_versi": loading_model_conv["metadata"]["svm_versi"] if loading_model_conv and "metadata" in loading_model_conv and "svm_versi" in loading_model_conv["metadata"] else None,
        "cnb_versi": loading_model_conv["metadata"]["cnb_versi"] if loading_model_conv and "metadata" in loading_model_conv and "cnb_versi" in loading_model_conv["metadata"] else None
        }), 200


@app.route('/predict/svm', methods=['POST'])
def predit_svm():
    try:
        text = validate_text_request(request.get_json(silent=True))
        logger.debug(f"Received text for prediction: {text}")
        transformed_text = tfidf_transform(text, tf_idf)
        prediction = svm.predict(transformed_text)
        return jsonify({'prediction': prediction.tolist()})
    except ValidationError as e:
        return jsonify({'error':e.message}), 400
    except Exception as e:
        # logger.exception(f"Error in /predict/svm: {str(e)}")
        logger.exception(f"Internal server error")
        return jsonify({'error': "Internal server error"}), 500

@app.route('/predict/naive-bayes', methods=['POST'])
def predict_naive_bayes():
    try:
        text = validate_text_request(request.get_json(silent=True))
        logger.debug(f"Received text for prediction: {text}")
        transformed_text = tfidf_transform(text, tf_idf)
        prediction = naive_bayes.predict(transformed_text)
        return jsonify({'prediction': prediction.tolist()})
    except ValidationError as e:
        return jsonify({'error':e.message}), 400
    except Exception as e:
        # logger.exception(f"Error in /predict/naive-bayes: {str(e)}")
        logger.exception(f"Internal server error")
        return jsonify({'error': "Internal server error"}), 500


if __name__ == '__main__':
    host = os.getenv('FLASK_HOST', '0.0.0.0')
    port = int(os.getenv('PORT', '5000'))
    serve(app, host=host, port=port)
