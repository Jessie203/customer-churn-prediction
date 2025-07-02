from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load model, scaler, and feature columns
model = joblib.load('app/model.pkl')
scaler = joblib.load('app/scaler.pkl')
feature_columns = joblib.load('app/columns.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    df = pd.DataFrame([data])

    # Map binary columns (assuming already converted if needed)
    binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map({'Yes': 1, 'No': 0})

    # One-hot encode
    df_encoded = pd.get_dummies(df, drop_first=True)

    # Align columns
    df_aligned = df_encoded.reindex(columns=feature_columns, fill_value=0)

    # Scale numeric
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    df_aligned[num_cols] = scaler.transform(df_aligned[num_cols])

    # Predict
    prediction = model.predict(df_aligned)[0]
    probability = round(model.predict_proba(df_aligned)[0][1], 2)

    # Example "LLM suggestion" idea
    if prediction == 1:
        message = "High risk of churn. Suggest offering a personalised retention plan."
    else:
        message = "Low churn risk. Recommend maintaining current engagement strategies."

    return jsonify({
        'PredictedChurn': int(prediction),
        'ChurnProbability': probability,
        'SuggestedAction': message
    })

if __name__ == '__main__':
    app.run(debug=True)
