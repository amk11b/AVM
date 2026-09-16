
import joblib
import pandas as pd
import numpy as np

def predict_property_value(property_data, model_dir="/Users/AVM/AVM/models"):
    """
    Predict property value using tiered model approach.

    property_data: dict with keys like 'bedrooms', 'latitude', 'longitude', etc.
    Returns: dict with predicted_price, confidence, comp_source
    """
    models = joblib.load(f"{model_dir}/tiered_models.joblib")
    features = joblib.load(f"{model_dir}/tiered_features.joblib")

    # Determine comp source and select appropriate model
    comp_source = property_data.get('comp_source', 'city')

    if comp_source == 'spatial' and models['spatial']:
        model = models['spatial']
        feature_list = features['spatial']
        confidence = 'high'
    elif comp_source == 'sector' and models['sector']:
        model = models['sector']
        feature_list = features['sector']
        confidence = 'medium'
    else:
        model = models['city']
        feature_list = features['city']
        confidence = 'low'

    # Prepare features
    X = pd.DataFrame([property_data])[feature_list].fillna(0)

    # Predict
    predicted_price = model.predict(X)[0]

    return {
        'predicted_price': predicted_price,
        'confidence': confidence,
        'comp_source': comp_source,
        'model_used': comp_source
    }
