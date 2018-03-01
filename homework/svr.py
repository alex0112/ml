#!/usr/bin/env python3


import pandas as pd
dataset = pd.read_csv('./data/svr/Salary_Data.csv')

X = dataset.iloc[:,  :1].values
y = dataset.iloc[:, 1:2].values


from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/4, random_state = 0)

## SVR:
from sklearn.svm import SVR
import numpy as np
regressor = SVR(kernel = 'rbf') ## rbf means use a Gaussian kernel
regressor.fit(X_train, y_train.ravel())

## Make prediction:
ypred = np.array(regressor.predict(X_test))

print(X_test, ypred)

import matplotlib.pyplot as plt
# plt.scatter(X_train, y_train, color='red')
# plt.scatter(X_test, y_test, color='blue')
plt.scatter(X_test, ypred, color='green')
# plt.plot(X_test, y_test, color='yellow')
plt.show()
