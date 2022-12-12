import time

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
import pickle
import time



app = Flask(__name__)
model = pickle.load(open("GaussianNbPickle.pkl", "rb"))

@app.route('/api', methods = ['GET'])
def returnascii():
    d = {}
    inputchr = str(request.args['query'])
    answer = str(ord(inputchr))
    d['output'] = answer + "ewan"
    return d

def response():
    query = dict(request.form)['query']
    result = query + " " + time.ctime()
    return jsonify({"response" : result})

@app.route('/predict', methods = ['POST'])
def predict():
    json_ = request.json
    query_df = pd.DataFrame(json_)
    prediction = model.predict(query_df)
    return jsonify({"Prediction": list(prediction)})
    # float_features = [float(x) for x in request.form.values()]
    # features = [np.array(float_features)]
    # prediction = model.predict(features)
    #return render template ("in")

if __name__ == "__main__":
    #app.run(host="0.0.0.0",)
    app.run(debug=True);

#comment