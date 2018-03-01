#!/usr/bin/env python3

import pandas as pd
dataset = pd.read_csv('./Position_Salaries.csv') ## Import the data.
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

## Feature Scaling
from sklearn.standard_scaler import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X    = sc_X.fit_transform(X)
y    = sc_y.fit_transform(y)

from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf') ## rbf means use a Gaussian kernel
regressor.fit(X, y)

ypred = regressor.predict(sc_X(np.array([[6.5]])))
ypred = sc_y.inverse_transform(ypred)


## Visualize the data:
plt.scatter(X, y, color='red')
plt.plot(X, regressor.predict(X), color='blue')
plt.title("Salary vs. XP (Training)")
plt.xlabel("XP")
plt.ylabel("$$$")
plt.show()

