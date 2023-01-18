# from pydataset import data
import pandas as pd
from sklearn.naive_bayes import GaussianNB
# from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from plotnine import *
from sklearn.pipeline import Pipeline
# from tensorflow import keras
from tensorflow.python import keras
# import keras
import tensorflow


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

df['outcome'].hist()

kf = KFold(n_splits=5)
print(f"\nkf is: \n{kf}\n")
# acc = []f



#Other forms for getting x and y:
# can be x = df.iloc[:,3:6].values
# y = df.iloc[:, -1].values


print(f"\ndf is: \n{df}\n")
print(f"\nX is: \n{X}\n")
print(f"\ny is: \n{y}\n")


#Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .20, random_state = 0)
#
# classifier = GaussianNB()
# classifier.fit(X_train, y_train)

print(f"\nX_train is: \n{X_train}\n")
print(f"\nX_test is: \n{X_test}\n")
print(f"\ny_train is: \n{y_train}\n")
print(f"\ny_test is: \n{y_test}\n")


#Training the model/Instantiating the model
clf = GaussianNB()
clf.fit(X_train, y_train)
#
# clf = LogisticRegression()
# clf.fit(X_train, y_train)


# #Model Summary
# print(f"\nModel Summary is: \n{clf.summary()}\n")

#Evaluating the model
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


#saving model
#pickle - to save model
#joblib - for large number of arrays

#Joblib to save the model
import joblib
joblib.dump(clf, "Naive")

#load the  model
rf = joblib.load("Naive")
print(f"\n rf is: \n {rf}\n")

#predicting using the model created
print(f"X_test is: \n{X_test}\n")
yPred = rf.predict(X_test)
print(f"rf.predict(X_test) is: \n{rf.predict(X_test)}\n")

#evaluate the loaded model
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
