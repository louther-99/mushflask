import time

import numpy as np
from flask import Flask, request, jsonify
import pickle
import time

app = Flask(__name__)

@app.route('/api', methods = ['GET'])
def returnascii():
    d = {}
    inputchr = str(request.args['query'])
    answer = str(ord(inputchr))
    d['output'] = answer
    return d

def response():
    query = dict(request.form)['query']
    result = query + " " + time.ctime()
    return jsonify({"response" : result})

if __name__ == "__main__":
    #app.run(host="0.0.0.0",)
    app.run(debug=True);