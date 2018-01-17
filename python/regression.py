import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression


## Get the data:
dataset = pd.read_csv('../moar_data/Salary_Data.csv') ## Import the data.
X = dataset.iloc[:, :-1].values  ## Select All of the values (i.e. not numbers or headers) and then remove the last column (X is the independent var)
y = dataset.iloc[:, -1].values ## Get the dependent var (the right-most column)

## Split the dataset into the training set, and test set.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

## Simple linear regression:
regressor = LinearRegression()
regressor.fit(X_train, y_train)  ## This means:  "From the data I have provided calculate the line of best fit to all of the points on the scatter plot"

## Make the prediction:
y_pred = regressor.predict(X_test)  ## This means: "Given an x value, where does that x value intersect with the line I fitted to this data (on the y axis) i.e. what's the y value of the point on the line of best fit that matches this x value?"

## Visualize the data:
plt.scatter(X_train, y_train, color='red')
plt.scatter(X_test, y_pred, color='green')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title("Salary vs. XP (Training)")
plt.xlabel("XP")
plt.ylabel("$$$")
plt.show()
