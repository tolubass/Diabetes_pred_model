from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import os
import pandas as pd
import logging
import sys
import openpyxl

logging.basicConfig(filename='diabetes.log', level=logging.DEBUG, format='%(asctime)s %(levelname)s: %(message)s')

sys.path.append(r'C:\Users\hp\Desktop\diabetes_folder_prediction')

# Loading the trained model
diabetes_model_path = r'C:\Users\hp\Desktop\diabetes_folder_prediction\diabetes_model2.pkl'
model = None

try:
    if os.path.exists(diabetes_model_path):
        with open(diabetes_model_path, 'rb') as model_file:
            model = pickle.load(model_file)
        logging.info("Model loaded successfully from diabetes_model2.pkl.")
        print("Model loaded successfully!")
        print("Expected input features:", model.n_features_in_)  # Debugging
    else:
        logging.error("Model file not found at the specified path.")
        print("Model file not found!")
except Exception as e:
    logging.error(f"Error loading model: {e}")
    print(f"Error loading model: {e}")

template_dir = r'C:\Users\hp\Desktop\diabetes_folder_prediction\templates'
excel_file_path = r'C:\Users\hp\Desktop\diabetes_folder_prediction\diabetes_predictions.xlsx'


app = Flask(__name__, template_folder=template_dir)

@app.route('/')
def home():
    logging.info("Home page accessed")
    return render_template("diabetes.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        logging.info("📩 Prediction request received")
        feature_names = ['Glucose', 'BMI', 'Age', 'DiabetesPedigreeFunction', 'BloodPressure']

        input_data = {feature: float(request.json.get(feature, 0)) for feature in feature_names}
        logging.debug(f"Inputs received: {input_data}")
        print("Received input:", input_data)  # Debug print

        input_features = np.array(list(input_data.values())).reshape(1, -1)

        if model is None:
            logging.error("Model is not loaded! Cannot proceed with prediction.")
            print("Model is not loaded!")
            return jsonify({'error': "Model failed to load. Check logs for details."})

        if input_features.shape[1] != model.n_features_in_:
            error_msg = f"Model expects {model.n_features_in_} features, but got {input_features.shape[1]}"
            logging.error(error_msg)
            print(error_msg)
            return jsonify({'error': error_msg})

        prediction = model.predict(input_features)[0]
        result = "Diabetic" if prediction == 1 else "Non-Diabetic"
        logging.info(f"Prediction result: {result}")

        data_to_save = {feature: [value] for feature, value in zip(feature_names, input_data.values())}
        data_to_save['Diabetes_Prediction_Result'] = [result]
        df = pd.DataFrame(data_to_save)

        if os.path.exists(excel_file_path):
            existing_df = pd.read_excel(excel_file_path)
            updated_df = pd.concat([existing_df, df], ignore_index=True)
            updated_df.to_excel(excel_file_path, index=False)
            logging.info("Data appended to existing Excel file.")
        else:
            df.to_excel(excel_file_path, index=False, engine = 'openpyxl')
            logging.info("New Excel file created and data saved.")

        return jsonify({'result': f'This patient is classified as {result}.'})

    except Exception as e:
        logging.error(f"Error occurred: {e}")
        print(f"Error occurred: {e}")  # Debug print
        return jsonify({'error': f"An error occurred: {e}"})

if __name__ == '__main__':
    logging.info("Starting Flask application...")
    print("\nFlask app running! Open your browser and go to: http://127.0.0.1:5000/\n")
    app.run(host='127.0.0.1', port=5000, debug=True)