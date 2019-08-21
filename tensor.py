# Learning Linear regression model with Python

import tensorflow
import keras
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import numpy as np
import pandas as pd
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style


data = pd.read_csv("student-mat.csv", sep=";")  # Here pd is panda and ; provides the separate data points.

data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]
# Labels(the fields we want to focus in entire data set) are the
# one we want to focus on from entire attributes(different fields in data set).

predict = "G3"

x = np.array(data.drop([predict], 1))  # here X is feature or attribute or variables.

y = np.array(data[predict])   # here Y is Label. y = mx + b (m = slope, b = y-axis intercept)
# in y = mx+b, m is 1 for 2-dimensional space but as we have 5 coefficient m is 5m(5 times m).

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split (x, y, test_size=0.1)

best = 0
for _ in range(30):   # we are gonna check if current score of model is better than previous for 30 times.
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split (x, y, test_size=0.1)
# these four are the variables and we have created four different arrays. sklearn will help data to get train.
# 0.1 means 10% of the entire data under training module.

    linear = linear_model.LinearRegression ()
    linear.fit (x_train, y_train)  # this line gonna find best fit line in x_train and y_train data).
    acc = linear.score (x_test, y_test)  # find accuracy of model.
    print (acc)

    if acc > best:
        best = acc    # now we will only save our model if current score of model is better than previous.
        with open("studentmodel.pickle", "wb") as f:   # here save your model use pickle.
            pickle.dump(linear, f)
            # to score highest scoring module so we don't have to retrain it again.
# wb is "write binary" - Python will overwrite the file, if it exists already or create it if the file doesn't exist.

pickle_in = open("studentmodel.pickle", "rb")  # rb = read binary
linear = pickle.load(pickle_in)         # this is gonna load our model variable called linear.


print('Coefficient: \n', linear.coef_)     # linear.coef_ will give us list of all coefficient.
print('Intercept: \n', linear.intercept_)  # this will give us y intercept.

#now based on this we will try to find out how much students gonna score.
predictions = linear.predict(x_test)

for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])
    # x_test[x] will show input data and y_test[x] will show actual value of final grade.

p = "studytime"   # G1, G2, studytime, failures, absences are our attribute(x) and G3 is our Label(y).
style.use("ggplot")
pyplot.scatter(data[p], data["G3"])
pyplot.xlabel(p)
pyplot.ylabel("Final Grade")   # here final grade is just G3 nothing else.
pyplot.show()


