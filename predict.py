from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import logging
import traceback
from pathlib import Path
from src.inference import InferencePipeline

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize Inference Pipeline
try:
    inference_pipeline = InferencePipeline()
    logger.info(f"Inference pipeline initialized using model at: {inference_pipeline.model_path}")
except Exception as e:
    logger.error(f"Failed to initialize inference pipeline: {e}")
    inference_pipeline = None


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy' if inference_pipeline is not None else 'unhealthy',
        'model_loaded': inference_pipeline is not None
    })


@app.route('/predict', methods=['POST'])
def predict():
    if inference_pipeline is None:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        # Handle single record or batch list
        if isinstance(data, dict):
            df_input = pd.DataFrame([data])
        else:
            df_input = pd.DataFrame(data)

        # Run inference using shared pipeline module
        results_df = inference_pipeline.predict(df_input)

        if isinstance(data, dict):
            record = results_df.iloc[0].to_dict()
            return jsonify(record)
        else:
            return jsonify(results_df.to_dict(orient='records'))

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)