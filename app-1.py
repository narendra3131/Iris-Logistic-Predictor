from flask import Flask, jsonify, render_template,request
import pickle
import os
import numpy as np


app = Flask(__name__)
model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
try:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    model = None
    print(f"Model file not found at {model_path}. Please provide the file.")
except Exception as e:
    model = None
    print(f"Error loading model: {e}")

# model = pickle.load(open('model.pkl','rb'))

target_names = ["setosa", "versicolor", "virginica"]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/pred',methods=['POST'])
def prediction():
    data = request.json
    try:
        sepalLength = float(data['sepalLength'])
        sepalWidth = float(data['sepalWidth'])
        petalLength = float(data['petalLength'])
        petalWidth = float(data['petalWidth'])
        features = np.array([[sepalLength, sepalWidth, petalLength, petalWidth]])
        print(features)
        prediction = model.predict(features)[0]
        return jsonify({"prediction": target_names[prediction]})
    except (KeyError, ValueError, TypeError):
        return "Invalid input", 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001,debug=True)