from flask import Flask, render_template, request, jsonify
import pickle
from model.feature_extraction import feature_gen

app = Flask(__name__)

# Load classifier
with open('model/intent_model.pkl', 'rb') as f:
    intent_model = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data['text']

    features = feature_gen(text)
    prediction = intent_model.predict(features)
    predicted_intent = prediction[0]

    return jsonify({'intent': predicted_intent})

if __name__ == '__main__':
    app.run(debug=True)