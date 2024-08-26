from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained RandomForestClassifier model
model = joblib.load('random_forest_model.pkl')

# Home route to render the HTML form
@app.route('/')
def index():
    return render_template('index.html')

# Prediction route to handle the POST request from the form
@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the request (JSON)
    data = request.get_json()

    # Convert the data into a DataFrame (matching the model's expected format)
    input_data = pd.DataFrame([data])

    # Make a prediction using the model
    prediction = model.predict(input_data)

    # Convert the prediction to standard Python int
    prediction_result = int(prediction[0])

    # Return the prediction as a JSON response
    l=[ 'Good',  'Moderate', 'Poor', 'Satisfactory', 'Severe', 'Very Poor']
    return jsonify({'prediction': l[prediction_result]})

if __name__ == '__main__':
    app.run(debug=True)
