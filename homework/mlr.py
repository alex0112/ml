#!/usr/bin/env python3 -W ignore::DeprecationWarning
import pandas as pd
import statsmodels.formula.api as sm
import numpy as np
from pprint import pprint
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression

## Get the data:
dataset = pd.read_csv('data/multiple_linear_regression/50_startups.csv')
X       = dataset.iloc[:, :-1].values
y       = dataset.iloc[:, -1].values

## Encode California, Florida, and NY as categorical variables
labelencoder_state = LabelEncoder()  
X[:, 3]            = labelencoder_state.fit_transform(X[:, 3])
hotencoder         = OneHotEncoder(categorical_features = [3])
X = hotencoder.fit_transform(X).toarray()

## Avoid the dummy variable trap.  (Leave off the first categorical variable, making the default be the first category)
X = X[:, 1:]

#####################################################################################################################
# LabelEncoder encodes the categories as 0, 1, 2, etc. (oridnal), OneHotEncoder takes those values and encodes them #
# as columns with 0 or 1 in them.                                                                                   #
# The regression object we are about to create applies "weights" (called b0, b1, b2 etc) to each column,            #
# the weight which is applied to each gets multiplied by the value of the column, meaning that the                  #
# weights for columns with 0 in them have no effect in the equation, whereas columns with a 1 do.                   #
#####################################################################################################################

## Split data into the training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

## Build and fit the Linear regression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

## Make predictions with the model
y_pred = regressor.predict(X_test)

## Backward eliminate:

## Choose a SL
SL = .05

###########################################################################
# Add a column of ones to our matrix of features (X)  this is because the #
# regressor library we're using ignores the b0 unless we give it          #
# something to multiply by.                                               #
###########################################################################

X = np.append(arr = np.ones(shape = (50, 1)).astype(int), values = X, axis = 1) ## Add a column of ones to our matrix of features (X)

## Build the optimal model:
X_opt = X[:, [:]]
regressor_OLS = sm.OLS(endog = y, exog = X).fit()
p_vals = regressor_OLS.pvalues[:]
highest_p_val = max(pvals)

while(highest_p_val > SL):
    

pprint(regressor_OLS.summary())
pprint(regressor_OLS.pvalues)

