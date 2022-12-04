# Import libraries
import numpy as np
from flask import Flask, request, jsonify
import pickle
import joblib

import firebase_admin
from firebase_admin import credentials, initialize_app

cred = credentials.Certificate("key.json")
default_app = firebase_admin.initialize_app(cred)




@app.route('/predict',methods=['POST'])
def predict():
    # Get the data from the POST request.
    data = request.get_json(force=True)
    # Make prediction using model loaded from disk as per the data.
    prediction = model.predict([[np.array(data['exp'])]])
    # Take the first value of prediction
    output = prediction[0]
    return jsonify(output)

def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = ''
    from .userAPI import userAPI
    app.register_blueprint(userAPI, url_prefix = '/user')

    # Load the model
    model = pickle.load(open('GaussianNbPickle.pkl', 'rb'))

app = create_app()
#Will run the app if main
if __name__ == '__main__':
    app.run(debug=True)
    # app.run(port=5000, debug=True)

#To make request to the server
# import requests
# url = 'http://localhost:5000/api'
# r = requests.post(url,json={'exp':1.8,})
# print(r.json())