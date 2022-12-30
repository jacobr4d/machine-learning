import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

#   DATASET
#   Simple Wines Data
#   Features: fixed acidity,volatile acidity,citric acid,residual sugar,chlorides,free sulfur dioxide,total sulfur dioxide,density,pH,sulphates,alcohol (continuous)
#   Target: quality (continuous)

wine_data_df = pd.read_csv('data/winequality_red.csv', sep=',', header=None)
wine_data_nd = wine_data_df[1:].to_numpy(np.float64)
X, y = wine_data_nd[:,:-1], wine_data_nd[:,[-1]]

#   ORDINARY LEAST SQAURES LINEAR REGRESSION BY HAND
#
#

# normalize data
X -= np.mean(X, axis=0)
X /= np.std(X, axis=0)

# add constant feature to fit a bias
X = np.insert(X,0,np.ones(len(X)),axis=1)

# Least Squares Weights
w_hand = np.linalg.inv(np.transpose(X) @ X) @ np.transpose(X) @ y

# Sum of Squares for this Model
y_h_hand = X @ w_hand
sos_hand = np.sum((y - y_h_hand) ** 2)
print(sos_hand)

#   VALIDATE RESULT BY COMPARING TO SKLEARN REGRESSION MODEL
#
#

# Show that this is equivalent to SK-Learn
regr = linear_model.LinearRegression()
regr.fit(X, y)
y_h_sklearn = regr.predict(X)
sos_sk = mean_squared_error(y, y_h_sklearn)
print(sos_sk * len(X))
assert(abs(sos_hand - sos_sk * len(X)) < 10 ** -5)
