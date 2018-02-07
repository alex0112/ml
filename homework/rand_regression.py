#!/usr/bin/env python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

TRAIN_DATA_PATH = './simple_linear_regression/train.csv'
TEST_DATA_PATH  = './simple_linear_regression/test.csv'

## Slurp the csv
train_data = pd.read_csv(TRAIN_DATA_PATH)
test_data  = pd.read_csv(TEST_DATA_PATH)

## Get the training and test values
X      = train_data.iloc[:, :-1].values
y      = train_data.iloc[:, -1].values
X_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values

## Create the regressor object from the training data
regressor = LinearRegression()
regressor.fit(X, y)

## Generate the predicted y values from the independent variables of the test set
y_prediction = regressor.predict(X_test)

## Visualize the data
plt.scatter(X, y, color='red')
plt.scatter(X_test, y_test, color='green')
plt.plot(X_test, y_prediction, color='blue')
plt.show()
