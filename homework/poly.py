#!/usr/bin/env python3

import pandas as pd
data = pd.read_csv('./data/polynomial_regression/harry_potter_revenue.csv')
X    = data.iloc[:, 2].values
y    = data.iloc[:, 0].values

# Split into training and test sets
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state = 0)

X_train = X_train.reshape(-1, 1) ## Because linear regressors expect 2D arrays.
y_train = y_train.reshape(-1, 1)


# ## Scale the data? 
# from sklearn.preprocessing import StandardScaler
# scaler_X = StandardScaler()
# X_train  = scaler_X.fit_transform(X_train)
# X_test   = scaler_x.fit(X_test)  ## Why not just scale before splitting?

## First fit a linear regression:
from sklearn.linear_model import LinearRegression
lin = LinearRegression()
lin.fit(X_train, y_train)

## Second fit a polynomial regressor
from sklearn.preprocessing import PolynomialFeatures
poly     = PolynomialFeatures(degree = 4) ## This is the itterations of the regressor?
X_poly   = poly.fit_transform(X_train)
poly_lin = LinearRegression().fit(X_poly, y_train)

import numpy as np
X_grid = np.arange(min(X_train), max(X_train), 0.1)  ## What does this do again?
#X_grid = X_grid.reshape(len(X_grid), 1)  ## I think it reorders the data and puts it back into a matrix.

# print(X)
# input()
# print(X_train)
# input()
# print(X_grid)
# input()
# X_grid = X_grid.reshape(len(X_grid), 1)  ## I think it reorders the data and puts it back into a matrix.
# print(X_grid)


import matplotlib.pyplot as plt
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, poly_lin.predict(poly.fit_transform(X_grid)), color = 'blue')
plt.title('Harry Potter Revenue over Time')
plt.xlabel('Weeks')
plt.ylabel('Revenue')
plt.show()
