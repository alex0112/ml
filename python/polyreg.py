#!/usr/bin/env python3
import numpy as np
import pandas as pd

DATA_PATH = './Position_Salaries.csv'
dataset   = pd.read_csv(DATA_PATH)
X         = dataset.iloc[:, 1:2].values  ## The upper bound is excluded in the 1:2.  This forces it to remain a matrix.
y         = dataset.iloc[:, 2].values    ## X Should always be a matrix, and y should always be a vector.

###############################################################################################################################
# Since the dataset is so small, we won't be splitting into a training and a test set in order to get as much out of our data #
# as we can.  We also don't need to scale.  (Not sure why this is)                                                            #
###############################################################################################################################

## Fitting a Linear regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

## Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 3)  ## The degree param is
X_poly   = poly_reg.fit_transform(X) # X_poly becomes a matrix of X^0, X^1, X^2
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly, y)


import matplotlib.pyplot as plt
## Visualize the linear regression
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary Level')
#plt.show()

## Visualize the polynomial regression
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, lin_reg2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary Level')
#plt.show()

## Making the prediction in a linear way
print(lin_reg.predict(6.5))
print("\n")
print(lin_reg2.predict(poly_reg.fit_transform(6.5)))
