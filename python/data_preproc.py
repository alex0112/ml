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

# Deal with undefined values by setting them to the average of the rest of the values in their column:
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0) # Create a new imputer object (this will process the data the way we want and turn it into something we like (see the next line))
imputer = imputer.fit(X[:, 1:3]) # Use the object we created to crunch the data we need and store it in that variable.  (Is this still an imputer object?)
X[:, 1:3] = imputer.transform(X[:, 1:3])  # 

## (Categorical variables (can be thought of as enums))
## Take values in columns that have a finite type set (i.e. are like "enums") and assign them to a number.  (E.g. Germany, France, and Spain are the only countries in this set, so make them 0, 1, and 2)
labelencoder_X = LabelEncoder() ## New LabelEncoder Obj
X[:, 0] = labelencoder_X.fit_transform(X[:, 0]) ## Fit the data, transform it (based upon the instructions in the LabelEncoder object we created?), and set the values in X

## Since it doesn't make sense to have Spain be zero, France be one, etc. (because we'll be doing math with those numbers later): We will instead make a new column for each
## enumerable type and have them be boolean (true if they are that country).  So each row will have a France column, a Spain column, and a Germany Column. If that row was a France row, those columns
## would be set to 1, 0, and 0 respectively.  (OneHotEncoder takes care of this for us)
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()

## The last columns are true or false, so they get encoded as categorical variables:
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y) # fit_transform is your friend.

## Split the dataset into the training set, and test set.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


## Scale features:
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

print(X)
print('****************************************************************************')
print(y)
print('****************************************************************************')
print(X_train, X_test)
print('****************************************************************************')
print(y_train, y_test)
