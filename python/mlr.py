#!/usr/bin/env python3 -W ignore::DeprecationWarning

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as sm
import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")


## Get the data:
dataset = pd.read_csv('/Users/alex/prog/ml/udemy_course/datafiles/Part 2 - Regression/Section 5 - Multiple Linear Regression/Multiple_Linear_Regression/50_Startups.csv') ## Import the data.  Duh. 
X = dataset.iloc[:, :-1].values  ## Select All of the values (i.e. not numbers or headers) and then remove the last column (X is the independent var)
y = dataset.iloc[:, 4].values ## Get the dependent var (the right-most column)

## The last columns are true or false, so they get encoded as categorical variables:
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [2])
X = onehotencoder.fit_transform(X).toarray()

## Avoiding the dummy var trap (We only need two dummy variables, one is the default
X = X[:, 1:]

## Split the dataset into the training set, and test set.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

"""## Scale features:
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
"""

## Fitting multiple linear regression 
regressor = LinearRegression()
regressor.fit(X_train, y_train)

## Predicting the Test set results
y_prediction = regressor.predict(X_test)

## Backward elimination:  (Eliminate less statistically significant variables)
# 1. Select a significance level
# 2. Fit the model with possible predictors
# 3. Consider the predictor with the highest P-Value (probability value) If P > SL (Probability Value > Significance Level) then go to step 4, otherwise go to FIN (meaning, finish)
# 4. Remove the variable
# 5. Go back to step three and fit the model without the variable (the predictor that failed the conditional)

## Build the model:  
X = np.append(
    arr = np.ones(
        shape = (50, 1)).astype(int),
    values = X,
    axis = 1
) ## Add a column of ones to our matrix of features (X)
#X = np.append(arr = np.ones(shape = (50, 1)).astype(int), values = X, axis = 1)  ## Add a column of ones to our matrix of features (X)

## Define the set of optimal features
import os
os.system('clear')

print("****************************************X*****************************************")
print(X,"\n\n")
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
print("****************************************X_opt*****************************************")
print(X_opt,"\n\n")

## 1. Select a significance level
SL = 0.05

## 2. Fit the model witht possible predictors
# Create a regressor object
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()

# 3. Consider the predictor with the highest P-Value (probability value) If P > SL (Probability Value > Significance Level) then go to step 4, otherwise go to FIN (meaning, finish)
p_vals = regressor_OLS.pvalues

print("****************************************p_vals*****************************************")
print(p_vals)
highest_p_val = max(p_vals)
predictors = X_opt

print("****************************************predictors*****************************************")
print(predictors, "\n\n")

print("\n\n\n\n\n\n")
# print("Backward Elimination:")
# while (highest_p_val > SL):
#     p_to_pred = dict(zip(p_vals, predictors))
#     highest_pred = p_to_pred[max(list(p_to_pred.keys()))]

#     print("&&&&&&&&&&&&&&&&&&&&&&&&&\n")
#     print(p_to_pred)
#     print("&&&&&&&&&&&&&&&&&&&&&&&&&\n")
#     print(highest_pred)
#     print("&&&&&&&&&&&&&&&&&&&&&&&&&\n")

#     ## New regressor
#     predictors = predictors.tolist().remove(highest_pred)
#     regressor_OLS = sm.OLS(endog = y, exog = predictors).fit()
#     p_vals = regressor_OLS.pvalues[:]
#     print(regressor_OLS.summary())
#     print(regressor_OLS.pvalues)
#     if (max(p_vals) > SL):
#         break

#     print(regressor_OLS.summary())
#     print(regressor_OLS.pvalues)
    
