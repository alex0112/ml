import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler

## Get the data:
dataset = pd.read_csv('../moar_data/Data.csv') ## Import the data.  Duh. 
X = dataset.iloc[:, :-1].values  ## Select All of the values (i.e. not numbers or headers) and then remove the last column (X is the independent var)
y = dataset.iloc[:, 3].values ## Get the dependent var (the right-most column)

## The last columns are true or false, so they get encoded as categorical variables:
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y) # fit_transform is your friend.

## Split the dataset into the training set, and test set.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


"""## Scale features:
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
"""
