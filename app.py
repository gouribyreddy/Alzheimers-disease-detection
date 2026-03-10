from flask import Flask, request, jsonify
import pickle
import numpy as np

# load trained model
model = pickle.load(open('alzheimers_model.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return "Alzheimer's Disease Prediction API"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['features']
    prediction = model.predict([np.array(data)])
    
    return jsonify({
        "prediction": int(prediction[0])
    })

if __name__ == '__main__':
    app.run(debug=True)
