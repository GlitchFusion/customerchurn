# importing modules
import os
import sys
import json
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_from_directory

# adding parent path to syspath
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.configs import Config
from src.utils.io_helpers import load_model, load_preprocessor
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# initializing flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'churn-prediction-temp-key'

# loading artifacts at startup
print("loading artifacts...")
preprocessor_data = load_preprocessor(Config.PREPROCESSOR_PATH)
custom_model = load_model(Config.CUSTOM_MODEL_PATH)

if preprocessor_data is None or custom_model is None:
    print("error: could not load preprocessor or model. please run main.py first.")

# getting preprocessor pipeline from loaded data
preprocessor = preprocessor_data['pipeline'] if preprocessor_data else None
feature_names = preprocessor_data['feature_names'] if preprocessor_data else None


# defining routes
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if preprocessor is None or custom_model is None:
        return jsonify({'error': 'model or preprocessor not loaded. run main.py first.'}), 500

    try:
        # extracting form data
        input_data = {
            'gender': request.form.get('gender'),
            'SeniorCitizen': int(request.form.get('SeniorCitizen', 0)),
            'Partner': request.form.get('Partner'),
            'Dependents': request.form.get('Dependents'),
            'tenure': int(request.form.get('tenure', 0)),
            'PhoneService': request.form.get('PhoneService'),
            'MultipleLines': request.form.get('MultipleLines'),
            'InternetService': request.form.get('InternetService'),
            'OnlineSecurity': request.form.get('OnlineSecurity'),
            'OnlineBackup': request.form.get('OnlineBackup'),
            'DeviceProtection': request.form.get('DeviceProtection'),
            'TechSupport': request.form.get('TechSupport'),
            'StreamingTV': request.form.get('StreamingTV'),
            'StreamingMovies': request.form.get('StreamingMovies'),
            'Contract': request.form.get('Contract'),
            'PaperlessBilling': request.form.get('PaperlessBilling'),
            'PaymentMethod': request.form.get('PaymentMethod'),
            'MonthlyCharges': float(request.form.get('MonthlyCharges', 0)),
            'TotalCharges': float(request.form.get('TotalCharges', 0))
        }

        # converting to dataframe
        input_df = pd.DataFrame([input_data])

        # creating engineered features same as preprocessorpy
        input_df['tenure_group'] = pd.cut(
            input_df['tenure'],
            bins=Config.TENURE_BINS,
            labels=Config.TENURE_LABELS
        )
        input_df['avg_monthly_spend'] = input_df['TotalCharges'] / (input_df['tenure'] + 1)

        # transform using preprocessor
        X_processed = preprocessor.transform(input_df)

        # predicting
        probability = custom_model.predict_proba(X_processed)[0]
        prediction = 1 if probability >= 0.5 else 0

        result = {
            'churn_probability': round(float(probability), 4),
            'churn_prediction': 'Yes' if prediction == 1 else 'No',
            'prediction_label': 'high risk' if prediction == 1 else 'low risk'
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/logs')
def logs():
    log_file = Config.LOG_PATH
    log_content = ""
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            log_content = f.read()
    else:
        log_content = "log file not found. please run main.py first."

    return render_template('logs.html', log_content=log_content)


@app.route('/plots')
def plots():
    figures_dir = Config.FIGURES_DIR
    plot_files = []
    if os.path.exists(figures_dir):
        for file in os.listdir(figures_dir):
            if file.endswith('.png'):
                title = file.replace('.png', '').replace('_', ' ').title()
                plot_files.append({
                    'filename': file,
                    'title': title,
                    'path': f'/plots/figures/{file}'
                })
        plot_files.sort(key=lambda x: x['filename'])
    else:
        plot_files = []

    return render_template('plots.html', plot_files=plot_files)


@app.route('/plots/figures/<filename>')
def serve_figure(filename):
    return send_from_directory(Config.FIGURES_DIR, filename)


@app.route('/metrics')
def get_metrics():
    metrics_file = os.path.join(Config.MODEL_DIR, 'metrics.json')
    
    if os.path.exists(metrics_file):
        try:
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
            return jsonify(metrics)
        except Exception as e:
            return jsonify({'error': f'error reading metrics file: {str(e)}'}), 500
    else:
        return jsonify({'error': 'metrics not found. please run main.py first.'}), 404


@app.route('/metrics/comparison')
def get_metrics_comparison():
    custom_metrics_file = os.path.join(Config.MODEL_DIR, 'metrics.json')
    sklearn_metrics_file = os.path.join(Config.MODEL_DIR, 'sklearn_metrics.json')
    
    result = {}
    
    if os.path.exists(custom_metrics_file):
        with open(custom_metrics_file, 'r') as f:
            result['custom'] = json.load(f)
    else:
        result['custom'] = {'error': 'custom metrics not found'}
    
    if os.path.exists(sklearn_metrics_file):
        with open(sklearn_metrics_file, 'r') as f:
            result['sklearn'] = json.load(f)
    else:
        result['sklearn'] = {'error': 'sklearn metrics not found'}
    
    return jsonify(result)


# running the app

if __name__ == '__main__':
    app.run(host=Config.API_HOST, port=Config.API_PORT, debug=True)