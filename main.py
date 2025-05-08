import numpy as np
import pandas as pd
import pickle
import os
import sys
from flask import Flask, request, jsonify

app = Flask(__name__)

# Configure error handling and logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    # Load the model and scaler
    logger.info("Loading model and scaler...")
    
    with open('best_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    logger.info(f"Model loaded successfully: {type(model)}")
    
    with open('scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    logger.info(f"Scaler loaded successfully: {type(scaler)}")
    
except Exception as e:
    logger.error(f"Error loading model or scaler: {str(e)}")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.json
        logger.info(f"Received data: {data}")
        
        # Create a DataFrame with the input data
        input_df = pd.DataFrame([data])
        
        columns_to_scale = ['Age', 'Fare']
        
        # Check if all required features are present
        required_features = ['Age', 'SibSp', 'Parch', 'Fare', 'Sex_male', 
                            'Embarked_Q', 'Embarked_S', 'Pclass_2', 'Pclass_3']
        
        missing_features = [feature for feature in required_features if feature not in input_df.columns]
        if missing_features:
            return jsonify({
                'error': f'Missing features: {missing_features}',
                'required_features': required_features,
                'provided_features': list(input_df.columns)
            }), 400
            
        
        X = input_df[required_features].copy()
        
        # Scale only the numeric columns that were scaled during training
        X_scaled = X.copy()
        X_scaled[columns_to_scale] = scaler.transform(X[columns_to_scale])
        logger.info("🔥"*40)
        
        # Log the final data being sent to the model
        logger.info(f"Processed input data shape: {X_scaled.shape}")
        logger.info(f"Processed input columns: {X_scaled.columns.tolist()}")
        
        # Make prediction
        prediction = model.predict(X_scaled)
        
        # Get probability if available
        probabilities = None
        try:
            probabilities = model.predict_proba(X_scaled)[0].tolist()
        except:
            pass
        
        # Return prediction
        response = {
            'prediction': int(prediction[0]),
            'survived': bool(prediction[0])
        }
        
        if probabilities:
            response['probability'] = probabilities
            
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)