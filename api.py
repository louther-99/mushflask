import json
import joblib
import numpy as np
from flask import Flask, request, jsonify
import pickle
import time
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.metrics import accuracy_score, confusion_matrix

print("Ho")
countShitYes = 0;
countShitNo = 0;

responsed = "";
acc = ['lightLevel', 'roomTemp', 'humidity'];

title = []
#Create an app usng the Flask class
app = Flask(__name__)

#Load the trained model (Pickle file)
model = pickle.load(open("Naive.pkl", "rb"))

dfo = pd.read_csv(r"datasets.csv")

reco = list()

print("Printing some shits")
lo = dfo['lightLevel'];
ro = dfo['roomTemp'];
ho = dfo['humidity'];
print(lo);
print(ho);
print(ro);
print("Done shitty")

print("Printing some shits")
llo = dfo['lightLevel'].tolist();
rro = dfo['roomTemp'].tolist();
hho = dfo['humidity'].tolist();
print(llo);
print(rro);
print(hho);
print("Done shitty")

for i in range(len(llo)):
    print(llo[i])
    reco.append(llo[i])
    reco.append(rro[i])
    reco.append(hho[i])

print(reco)

# for x in ll.length:
#     lll = ll[x] + hh[x] + rr[x]
# print(lll);

print(f" Dataframe Head: \n {dfo.head()}\n")
print(f" Dataframe Described: \n {dfo.describe()}\n")
mushroom_featureso = ['lightLevel', 'roomTemp', 'humidity']
mushroom_classo = ['outcome']
Xo = dfo[mushroom_featureso]
yo = dfo[mushroom_classo]

# dfo['outcome'].hist()

kfo = KFold(n_splits=5)
print(f"\nkf is: \n{kfo}\n")
# acc = []f

# Other forms for getting x and y:
# can be x = df.iloc[:,3:6].values
# y = df.iloc[:, -1].values

print(f"\ndf is: \n{dfo}\n")
print(f"\nX is: \n{Xo}\n")
print(f"\ny is: \n{yo}\n")

# Split data
X_train, X_test, y_train, y_test = train_test_split(Xo, yo, test_size=.20, random_state=0)
#
# classifier = GaussianNB()
# classifier.fit(X_train, y_train)

print(f"\nX_train is: \n{X_train}\n")
print(f"\nX_test is: \n{X_test}\n")
print(f"\ny_train is: \n{y_train}\n")
print(f"\ny_test is: \n{y_test}\n")

# Training the model/Instantiating the model
clfo = GaussianNB()
clfo.fit(X_train, y_train)
#
# clf = LogisticRegression()
# clf.fit(X_train, y_train)

# #Model Summary
# print(f"\nModel Summary is: \n{clf.summary()}\n")

# Evaluating the model
y_predo = clfo.predict(X_test)
ac = accuracy_score(y_test, y_predo);

@app.route('/', methods = ['POST', 'GET'])
def maine():
    # df = pd.read_csv(r"/Users/loutherolayres/PycharmProjects/mush/MyDataSetCSV2.csv")
    df = pd.read_csv(r"datasets.csv")

    rec = list()

    print("Printing some shits")
    l = df['lightLevel'];
    r = df['roomTemp'];
    h = df['humidity'];
    print(l);
    print(h);
    print(r);
    print("Done shitty")

    print("Printing some shits")
    ll = df['lightLevel'].tolist();
    rr = df['roomTemp'].tolist();
    hh = df['humidity'].tolist();
    print(ll);
    print(rr);
    print(hh);
    print("Done shitty")

    for i in range(len(ll)):
        print(ll[i])
        rec.append(ll[i])
        rec.append(rr[i])
        rec.append(hh[i])

    print(rec)

    # for x in ll.length:
    #     lll = ll[x] + hh[x] + rr[x]
    # print(lll);

    print(f" Dataframe Head: \n {df.head()}\n")
    print(f" Dataframe Described: \n {df.describe()}\n")
    mushroom_features = ['lightLevel', 'roomTemp', 'humidity']
    mushroom_class = ['outcome']
    X = df[mushroom_features]
    y = df[mushroom_class]

    # df['outcome'].hist()

    kf = KFold(n_splits=5)
    print(f"\nkf is: \n{kf}\n")
    # acc = []f

    # Other forms for getting x and y:
    # can be x = df.iloc[:,3:6].values
    # y = df.iloc[:, -1].values

    print(f"\ndf is: \n{df}\n")
    print(f"\nX is: \n{X}\n")
    print(f"\ny is: \n{y}\n")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.20, random_state=0)
    #
    # classifier = GaussianNB()
    # classifier.fit(X_train, y_train)

    print(f"\nX_train is: \n{X_train}\n")
    print(f"\nX_test is: \n{X_test}\n")
    print(f"\ny_train is: \n{y_train}\n")
    print(f"\ny_test is: \n{y_test}\n")

    # Training the model/Instantiating the model
    clf = GaussianNB()
    clf.fit(X_train, y_train)
    #
    # clf = LogisticRegression()
    # clf.fit(X_train, y_train)

    # #Model Summary
    # print(f"\nModel Summary is: \n{clf.summary()}\n")

    # Evaluating the model
    y_pred = clf.predict(X_test)
    ac = accuracy_score(y_test, y_pred);
    print(f"Accuracy Score: \n {accuracy_score(y_test, y_pred)}\n")
    print(f"Classification Report: \n {classification_report(y_test, y_pred)}\n")

    print('Printing shits')
    print(y_pred)
    print("\n")
    print(y_test)
    print("\n")
    print(ac)

    # saving model
    # pickle - to save model
    # joblib - for large number of arrays

    # Joblib to save the model
    import joblib
    joblib.dump(clf, "Naive")

    # load the  model
    rf = joblib.load("Naive")
    print(f"\n rf is: \n {rf}\n")

    # predicting using the model created
    print(f"X_test is: \n{X_test}\n")
    yPred = rf.predict(X_test)
    print(f"rf.predict(X_test) is: \n{rf.predict(X_test)}\n")

    # evaluate the loaded model
    acc = accuracy_score(y_test, yPred)
    print(f"\nAccuracy: \n{acc}\n")
    cm = confusion_matrix(y_test, yPred)
    print(f"\nConfusion Matrix: \n{cm}\n")

    print(f"\nModel Score: \n{clf.score(X_test, y_test)}\n")
    print(metrics.classification_report(y_test, yPred))
    # print(mean_absolute_error(y_test, yPred))
    # msa = mae(y_test, yPred)
    # print("\nPrinting MAE:\n")
    # print(mae(y_test, yPred))
    # print("Mean absolute error : " + str(msa))
    # print(f"msa is: \n{msa}\n")

    # mean_absolute_error(y_test, yPred)

    import pickle
    pickle.dump(clf, open('Naive.pkl', "wb"))

    # loaded_pickle_model = pickle.load(open("GaussianNbPickle.pkl"),"rb")

    # pickle_y_preds = loaded_pickle_model.predict(X_train)
    # evaluate_preds(y_test, pickle_y_preds)
    #
    # #keras
    # kerasFile = 'naive.h5';
    # keras.models.save_model(clf, kerasFile)
    #

    # #Converting the model to Tf model
    # new_model = keras.models.load_model('GaussianNbPickle')
    # converter = tensorflow.lite.TFLiteConverter.from_keras_model(new_model)
    # tflite_model = converter.convert()
    # open("converted_tf_lite.tflite", "wb").write(tflite_model)
    #
    #

    # from sklearn.preprocessing import LabelEncoder
    # import numpy as np
    # #read csv data
    # import pandas as pd
    # df = pd.read_csv(r"/Users/loutherolayres/PycharmProjects/mush/MyDataSetCSV2.csv")
    # # x = df.iloc[:,3:6].values
    # # y = df.iloc[:, -1].values
    #
    # #encoding the strings to Numericals
    # Numerics = LabelEncoder()
    #
    # #Dropping the targer variable and making it as a newframe
    # cols_to_drop = ['id', 'batchNumber', 'datetime']
    # df = df.drop(cols_to_drop, axis = 1)
    # print(df.describe())
    # #prints 8 numbers // std is the standard deviation, which measures how numerically spread out the values are.
    # print('Done Describing\n\n')
    # print(f"Columns are: \n {df.columns}")
    # print("\nDone printing columns \n")
    #
    # y = df.outcome
    # print(f"y is: \n\n {y} \nDone printing y = df.outcome")
    #
    # #assigning features
    # mushroom_features = ['lightLevel', 'roomTemp', 'humidity']
    # x = df[mushroom_features]
    #
    # #describing x
    # print(f"\nDescribing x: \n{x.describe()} \n")
    #
    # #printing x.head
    # print(f"\nx.head is: \n {x.head()}\n")
    #
    # #Printing Dataframe
    # print(f"\nDataframe is: \n {df}")
    #
    # #Create the model
    #
    # #Mean absolute error
    # # predicted_home_prices = melbourne_model.predict(X)
    # # mean_absolute_error(y, predicted_home_prices)
    #
    # # Splitting data
    # # from sklearn.model_selection import train_test_split
    # #
    # # # split data into training and validation data, for both features and target
    # # # The split is based on a random number generator. Supplying a numeric value to
    # # # the random_state argument guarantees we get the same split every time we
    # # # run this script.
    # # train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)
    # # # Define model
    # # melbourne_model = DecisionTreeRegressor()
    # # # Fit model
    # # melbourne_model.fit(train_X, train_y)
    # #
    # # # get predicted prices on validation data
    # # val_predictions = melbourne_model.predict(val_X)
    # # print(mean_absolute_error(val_y, val_predictions))
    #
    # # print("Making predictions for the following data:")
    # # print(x.head())
    # # print("The predictions are")
    # # print(df.predict(x.head()))
    #
    # #
    # # inputs = df.drop('outcome', axis = 'columns')
    # # target = df['outcome']
    # # print(target)
    #
    # # #creating the new dataframe
    # # inputs['lightLevel_n'] = Numerics.fit_transform(inputs['lightLevel'])
    # # inputs['roomTemp_n'] = Numerics.fit_transform(inputs['roomTemp'])
    # # inputs['humidity_n'] = Numerics.fit_transform(inputs['humidity'])
    # # print(inputs)
    # #
    #
    # #preprocessing
    # #train test split
    # # from sklearn.model_selection import train_test_split
    # # x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=10)
    #
    # # print(x.shape)
    # # print(x_train.shape)
    # # print(x_test.shape)
    #
    # #training set
    # #validation set
    # #test set
    #
    # # Main set (Train and validation)
    # # and test set
    #
    # #train test validation split
    # # from sklearn.model_selection import train_test_split
    # # x_main, x_test, y_main, y_test = train_test_split(x,y,test_size=10)
    # # x_train, x_val, y_train, y_val = train_test_split(x_main, y_main, test_size=10)
    #
    # #scaling our data in order to not to be deceived by the scale of the columns to bring our features to similar scales
    # # print(x_train) #
    #
    # #1 is min max scaler
    # #2 is
    #
    # #scaling
    # #x is data
    # """
    # x' = (x-min) / (max-min)
    # """
    # # from sklearn.preprocessing import MinMaxScaler
    # # scaler = MinMaxScaler()
    # # x_train = scaler.fit_transform(x_train) # updating x_train
    # # x_tests = scaler.transform(x_test)
    # # print(x_train)
    #
    # """
    # x' = (x-mean) / std
    # """
    # # from sklearn.preprocessing import StandardScaler
    # # scaler = StandardScaler() #performs better than min max
    # # x_train = scaler.fit_transform(x_train)
    # # y_test = scaler.transform(x_test)
    # # print(x_train)
    # #
    # # #KNN algo
    # # from sklearn.neighbors import KNeighborsClassifier
    # # model = KNeighborsClassifier(n_neighbors=2)
    # # model.fit(x_train, y_train)
    # #
    # # #prediction
    # # y_pred = model.predict(x_tests)
    # #
    # # #evaluation
    # # from sklearn.metrics import accuracy_score
    # # acc = accuracy_score(y_test, y_pred)
    # # print(f"Accuracy: {acc}")
    # #
    # #
    # # #confusion matrix
    # # from sklearn.metrics import confusion_matrix
    # # cm = confusion_matrix(y_test, y_pred, labels=[0,1])
    # # print("Confusion matrix: ")
    # # print(cm)
    #
    #
    #
    # print("\n\nStart\n")
    #
    # # Importing the libraries
    # import numpy as np
    # import matplotlib.pyplot as plt
    # import pandas as pd
    #
    # # Importing the dataset
    # dataset = pd.read_csv('/Users/loutherolayres/PycharmProjects/mush/MyDataSetCSV2.csv')
    # X = dataset.iloc[:,3:6].values
    # y = dataset.iloc[:, -1].values
    # print(f"\nX is: \n{X} \n")
    # print(f"\nyis: \n{y} \n")
    #
    # # Splitting the dataset into the Training set and Test set
    # from sklearn.model_selection import train_test_split
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
    # print(f"\nX_train is: \n{X_train} \n")
    # print(f"\nX_test is: \n{X_test} \n")
    # print(f"\ny_train is: \n{y_train} \n")
    # print(f"\ny_test is: \n{y_test} \n")
    #
    # # Feature Scaling
    # from sklearn.preprocessing import StandardScaler
    # sc = StandardScaler()
    # print(f"\n sc is: \n{sc} \n")
    # X_train = sc.fit_transform(X_train)
    # X_test = sc.transform(X_test)
    # print(f"\nX_train is: \n{X_train} \n")
    # print(f"\nX_test is: \n{X_test} \n")
    #
    # # Training the Naive Bayes model on the Training set
    # from sklearn.naive_bayes import GaussianNB
    # classifier = GaussianNB()
    # classifier.fit(X_train, y_train)
    #
    # # Predicting the Test set results
    # y_pred = classifier.predict(X_test)
    # print(f"y_pred: \n {y_pred}\n")
    #
    # # Making the Confusion Matrix
    # from sklearn.metrics import confusion_matrix, accuracy_score
    # ac = accuracy_score(y_test,y_pred)
    # cm = confusion_matrix(y_test, y_pred)
    # print(f"Acc: \n{ac}\n")
    # print(f"CM: \n{cm}\n")
    #

    return jsonify({"Accuracy" : acc });


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

@app.route('/descriptionByBatch', methods = ['GET', 'POST'])
def getDes():
    des2jsns = request.json
    dfs = pd.read_csv(r"my8.csv")
    print(f" Dataframe Head: \n {dfs.head()}\n")
    print(f" Dataframe Described: \n {dfs.describe()}\n")
    dess = dfs.describe()
    desss = dess.to_json()
    dess2Dicts = dess.to_dict()
    print(f"dess: {dess}")
    print(f"desss: {desss}")
    print(f"des2jsns: {des2jsns}");
    print("Dess before return")
    # return desss
    #dess.to_json() then simply return
    # return jsonify({"Response" : des2jsns, "Responde" : dess2Dicts})
    return jsonify({"Responde" : dess2Dicts, "Response" : des2jsns, })




@app.route('/convertByBatch', methods = ['POST'])
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
    dfItem.to_csv('my8.csv', index = False)
    print(dfItem);
    print(dfItem.head())
    des = dfItem.describe()
    # sonify = jsonify(des)
    # desDump = json.dumps(des)
    print(f"Des: {des}")
    jsonDes = des.to_json()
    print(f"Des to json {jsonDes}")
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

    df = pd.read_csv("my8.csv")

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
    print("Shifties")
    print(countShitYes);
    print(countShitNo);

    # jsonify({"Description": jsonDes})

    # return (now, acsjson)
    # return (jsonDes + jsonify({"Prediction": newJsn, "Accuracy" : acs, "Outcome" : determiner }))
    # return  jsonify({"Description" : jsonDes})
    return jsonify({"Prediction": newJsn, "Accuracy" : acs, "Outcome" : determiner, "Yes" : countShitYes, "No" :  countShitNo});
    #
    # value = {
    #     "Prediction": newJsn,
    #     "Accuracy": acs,
    #     "Outcome" : determiner,
    #     # "Description" : jsonify(jsonDes)
    # }
    # val = json.dumps(value)
    # # return (val,jsonDes)
    # # return jsonDes
    # return val


@app.route('/description2ByIndiv', methods = ['GET', 'POST'])
def getDes2():

    desss2Json = request.json
    dfs = pd.read_csv(r"datasets.csv")
    print(f" Dataframe Head: \n {dfs.head()}\n")
    print(f" Dataframe Described: \n {dfs.describe()}\n")
    dess2 = dfs.describe()
    desss = dess2.to_json()
    dess2Dict = dess2.to_dict()
    print(f"dess: {desss}")
    print(f"dess2Dict: {dess2Dict}")
    print(f"des2jsns: {desss2Json}");
    print("Dess before return")
    # return desss
    #dess.to_json() then simply return
    # return jsonify({"Response" : des2jsns, "Responde" : dess2Dicts})
    return jsonify({"Responde" : dess2Dict, "Response" : desss2Json, })



@app.route('/predictByIndiv', methods = ['POST', 'GET']) #POST To send data
def predict():
    print("Start of predictByIndiv")
    # main.read()
    di = []
    # Receive parameter from json
    # j = request.get_json()
    # print(j);
    des2jsn = request.json
    df = pd.read_csv(r"datasets.csv")
    dess2 = df.describe()
    dess2Dict = dess2.to_dict()
    desss2Json = dess2.to_json()
    print(f"dess2Dict: {dess2Dict}")
    print(f"desss2Json: {desss2Json}")
    print(f"des2jsn: {des2jsn}");
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

    lightLevel = float(json_['lightLevel'])
    roomTemp = float(json_['roomTemp'])
    humidity = float(json_['humidity'])

    features = [lightLevel, roomTemp, humidity]
    print("\nPrinting lightlevel, roomtemp, humidity");
    print(lightLevel, roomTemp, humidity)

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
    # return jsonify({"Prediction": list(prediction), "Accuracy" : ac, "Responde" : dess2Dict, "Response" : des2jsn});
    return jsonify({"Prediction": list(prediction), "Accuracy" : ac, "Responde" : dess2Dict});
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
# import requests
# url = 'http://localhost:5000/api'
# r = requests.post(url,json={'exp':1.8,})
# print(r.json())