import json
import time
from main import y_test, y_pred, ac
import joblib
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import time

print("Ho")
countShitYes = 0;
countShitNo = 0;

responsed = "";
acc = ['lightLevel', 'roomTemp', 'humidity'];

title = []
#Create an app usng the Flask class
app = Flask(__name__)

#Load the trained model (Pickle file)
model = pickle.load(open("GaussianNbPickle.pkl", "rb"))

@app.route('/api', methods = ['GET'])
def returnascii():
    d = {}
    inputchr = str(request.args['query'])
    answer = str(ord(inputchr))
    d['output'] = answer + "ewan"
    return d

#com/api?query=2
def response():
    query = dict(request.form)['query']
    result = query + " " + time.ctime()
    return jsonify({"response" : result})

#route() decorator to tell Flask what URL should trigger our function

@app.route('/convert', methods = ['POST'])
def getJsontoCsv(countShitYes = 0, countShitNo = 0):
    jsn = request.json
    # df2 = pd.read_json(jsn, orient='index')
    # print("Printing df2")
    # print(df2)
    print("Printing jsn")
    print(jsn)
    print("Done printing jsn")
    print(jsn[0]['outcome']); #no
    # title = ('title' + {jsn['title']});
    # data = ('data' + {jsn['data']});
    #
    # t = jsn['title']
    # d = jsn['data']
    # for data in d:
    #     y = data
    #
    # print(t)
    # print(d)
    # print(y)
    # datas = title + data
    # print('Datas:')
    # print(datas)

    # s = jsonify(jsn)
    # print(s)
    # return jsonify({"response" : jsn})
    # dat = json.loads()
    # print("Printing dat")
    # print(dat)
    # jsonD = json.dumps(jsn)
    # jsons = json.loads(jsn)
    # print(jsonD);
    # print("Done printing jsonD")
    # print(jsonD[0])
    # print(jsonD[1])
    # sx = json.loads(jsn)
    # print("\njsons: ");
    # print(sx);
    # new = pd.read_json(jsn)
    # new.to_csv('wakaruu.csv');

    # rf = joblib.load("GaussianNb")
    # yPred = rf.predict(X_test)
    # print(f"rf.predict(X_test) is: \n{rf.predict(X_test)}\n")

    dfItem = pd.DataFrame.from_records(jsn)
    dfItem.to_csv('my7.csv', index = False)
    print(dfItem);
    print(dfItem.head())
    print(dfItem.describe());

    # l = dfItem['lightLevel'];
    # r = dfItem['roomTemp'];
    # h = dfItem['humidity'];
    #
    # print(l);
    # print(h);
    # print(r);


    # df = pd.read_json(jsons)
    # df.to_csv('my3.csv')



    lightLevel = float(jsn[0]['lightLevel'])
    roomTemp = float(jsn[0]['roomTemp'])
    humidity = float(jsn[0]['humidity'])

    print("Printing length of json")
    # json_string = json.dumps(jsn)
    # byte_ = json_string.encode("utf-8")
    size_in_bytes = len(jsn)
    print(size_in_bytes)
    list = []
    listpred = []
    newJsn = [];
    newJsn = jsn;

    print(f"New json is {newJsn}")
    print(f"json is {jsn}\n")

    df = pd.read_csv("my7.csv")

    for x in range(size_in_bytes):
        lightLevel = float(jsn[x]['lightLevel'])
        roomTemp = float(jsn[x]['roomTemp'])
        humidity = float(jsn[x]['humidity'])
        print (lightLevel, roomTemp, humidity)
        featuress = [lightLevel, roomTemp, humidity]
        list.append(featuress)

        featuresss = [np.array(list[x])]
        print("printing featuresss")
        print(featuresss)
        prediction = model.predict(featuresss)
        print(f"Prediction iss {prediction}")
        print(f"jsn[{x}]['outcome'] is {jsn[x]['outcome']}")
        # print(str(prediction))
        listpred.append(prediction)
        if (prediction == 'Yes'):
            print("Nasa yes")
            # newJsn[x]['outcome'] = "Yes"
            newJsn[x]['outcome'] = newJsn[x]['outcome'].replace('No', 'Yes')
            print(f"newJsn[{x}]['outcome'] is now {newJsn[x]['outcome']}")
            countShitYes += 1
        if (prediction == 'No'):
            print("Nasa no")
            # newJsn[x]['outcome'] = "No"
            newJsn[x]['outcome'] = newJsn[x]['outcome'].replace("Yes", 'No')
            print(f"newJsn[{x}]['outcome'] is now {newJsn[x]['outcome']}")
            countShitNo += 1

        print(f"jsn[{x}]['outcome'] is {jsn[x]['outcome']}")
        print(f"jsn[{x}]['outcome'] is {jsn[x]['outcome']}")
        print(f"Then {jsn[x]['outcome']}")
        print(f"Now {newJsn[x]['outcome']}")



        # updating the column value/data
        if(listpred[x] == ['Yes']):
            y = 'Yes'
            df.loc[x, 'outcome'] = y

        if (listpred[x] == ['No']):
            n = 'No'
            df.loc[x, 'outcome'] = n


    df['outcome'] = df['outcome'].replace("['Yes']", 'Yes')
    df['outcome'] = df['outcome'].replace("['No']", 'No')


    print(f"After the loop json: {jsn}")
    print(f"After the loop newJsn: {newJsn}")



    total = countShitYes + countShitNo
    determiner = round((countShitYes / total) * 100)
    print(f"countShitYes: {countShitYes}")
    print(f"countShitNo: {countShitNo}")
    print(f"total: {total}")
    print(f"determiner: {determiner}")

    # writing into the file
    df.to_csv("AllD.csv", index=False)
    # sh = str(listpred)
    # ch =
    # list_of_str = [elem.replace(ch, 'Yes') for elem in sh]
    # # new = listpred.replace(['Yes'], 'Yes')
    # print(list_of_str)
    print("listpred")
    print(listpred)
    print("listpred[0]")
    print(listpred[0])
    print("list")
    print(list)
    print("list[1]")
    print(list[1])
    print("Printing another shits")
    print(lightLevel);
    print(roomTemp);
    print(humidity);
    print("Done printing another shits")



    # print("Done printing df pd.read_json")
    # df = pd.DataFrame(jsn)
    # print(df)
    # print(f" Dataframe Head: \n {df.head()}\n")
    # print(f" Dataframe Described: \n {df.describe()}\n")
    # mushroom_featuress = ['lightLevel', 'roomTemp', 'humidity']
    # mushroom_classs = ['outcome']
    # X = df[mushroom_featuress]
    # y = df[mushroom_classs]
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.20, random_state=0)
    #
    # clfs = GaussianNB()
    # clfs.fit(X_train, y_train)
    #
    # y_pred = clfs.predict(X_test)
    # acR = accuracy_score(y_test, y_pred);
    # print(f"Accuracy Score: \n {acR}\n")
    # print(f"Accuracy Score: \n {accuracy_score(y_test, y_pred)}\n")
    # print(f"Classification Report: \n {classification_report(y_test, y_pred)}\n")
    # return jsonify(jsn)


    print("Starting to create a new model");
    dfwutt = pd.read_csv(r"AllD.csv")
    print(f" Dataframe Head: \n {dfwutt.head()}\n")
    print(f" Dataframe Described: \n {dfwutt.describe()}\n")

    # for row in dfItem.loc[row, 'lightLevel']

    mu_features = ['lightLevel', 'roomTemp', 'humidity']
    mu_class = ['outcome']
    XXX = dfwutt[mu_features]
    yyy = dfwutt[mu_class]
    kff = KFold(n_splits=5)
    print(f"\nkf is: \n{kff}\n")
    print(f"\ndfwut is: \n{dfwutt}\n")
    print(f"\nX is: \n{XXX}\n")
    print(f"\ny is: \n{yyy}\n")

    # Split data
    XX_train, XX_test, yy_train, yy_test = train_test_split(XXX, yyy, test_size=.20, random_state=0)
    print(f"\nXX_train is: \n{XX_train}\n")
    print(f"\nXX_test is: \n{XX_test}\n")
    print(f"\nyy_train is: \n{yy_train}\n")
    print(f"\nyy_test is: \n{yy_test}\n")

    clfss = GaussianNB()
    print("Fitting")
    clfss.fit(XX_train, np.ravel(yy_train))


    yy_preds = clfss.predict(XX_test)
    acs = accuracy_score(yy_test, yy_preds);

    print(acs);

    print(f"Accuracy Score: \n {accuracy_score(yy_test, yy_preds)}\n")
    print(f"Classification Report: \n {classification_report(yy_test, yy_preds)}\n")

    cmm = confusion_matrix(yy_test, yy_preds)
    print(f"\nConfusion Matrix: \n{cmm}\n")

    print("Trying to save a new model")
    joblib.dump(clfss, "GaussianNbV2shitsss")
    print("Done creating a model");

    print("Printing jsonify(jsn) before returning")
    wut = jsonify(jsn)
    now = jsonify(newJsn)
    acsjson = jsonify(acs)
    print(wut)
    print(now)
    # return (now, acsjson)
    return jsonify({"Prediction": newJsn, "Accuracy" : acs, "Outcome" : determiner });

@app.route('/predict', methods = ['POST']) #POST To send data
def predict():
    # main.read()
    di = []
    # Receive parameter from json
    # j = request.get_json()
    # print(j);

    df = pd.read_csv(r"/Users/loutherolayres/PycharmProjects/mush/MyDataSetCSV2.csv")
    described = df.describe();
    print(f" Dataframe Described: \n {df.describe()}\n")
    mushroom_features = ['lightLevel', 'roomTemp', 'humidity']
    mushroom_class = ['outcome']
    X = df[mushroom_features]
    y = df[mushroom_class]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.20, random_state=0)

    clf = GaussianNB()
    clf.fit(X_train, y_train)


    json_ = request.json
    print(json_);
    print("Done printing json\n")
    # python_data = json.loads(json_)
    # print(python_data)
    # print("Done printing python_data")

    # lightLevel = int(python_data['lightLevel'])
    # roomTemp = int(python_data['roomTemp'])
    # humidity = int(python_data['humidity'])

    lightLevel = float(json_['lightLevel'])
    roomTemp = float(json_['roomTemp'])
    humidity = float(json_['humidity'])

    features = [lightLevel, roomTemp, humidity]
    print("\nPrinting lightlevel, roomtemp, humidity");
    print(lightLevel, roomTemp, humidity)
    # return (humidity)
    print("\nPrinting features below")
    print(features)

    featuress = [np.array(features)]
    print("\nprinting featuress")
    print(featuress)
    prediction = model.predict(featuress)
    print("\nPrinting y_test")
    # print(y_test)

    # ac = accuracy_score(y_test, y_pred);
    # ac = accuracy_score(y_test, prediction);
    # print(f'\n{ac} is ac')
    print(jsonify({"Prediction": list(prediction)}))
    return jsonify({"Prediction": list(prediction), "Accuracy" : ac});
    # return ({"Prediction": list(prediction)});
    # return jsonify({"Prediction": list(prediction)});
    # return ("Response" + prediction)
    # return jsonify(features);


    # query_df = pd.DataFrame(json_)
    # di.append(json_['lightLevel'])
    # di.append(json_['roomTemp'])
    # di.append(json_['humidity'])
    # print(di);
    # print('done printing di');
    # prediction = model.predict(query_df)
    # return jsonify({"Prediction": list(prediction)});

    # prediction = model.predict(di)
    # return (json_);
    # return prediction;


    # json_ = request.json
    # query_df = pd.DataFrame(json_)
    # print(query_df + "is query_df");
    # prediction = model.predict(query_df)
    # return jsonify({"Prediction": list(prediction)});


    # float_features = [float(x) for x in request.form.values()]
    # features = [np.array(float_features)]
    # prediction = model.predict(features)
    #return render template ("in")

@app.route('/try', methods = ['GET', 'POST'])
def trys():
    global responsed;
    if(request.method == 'POST'):
        request_data = request.data
        request_data = json.loads(request_data.decode())
        batchNumber = request_data['batchNumber'];
        responsed = f'Hi  {batchNumber}! this is batch Python';
    else:
        return jsonify({'batchNumber': responsed});
@app.route('/print', methods = ['GET'])
def hello_world():
    return 'Hello, World!'

print("Before run")

if __name__ == "__main__":
    print("Run")
    #app.run(host="0.0.0.0",)
    app.run(debug=True);

#comment