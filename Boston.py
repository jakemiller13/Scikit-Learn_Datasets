# -*- coding: utf-8 -*-
"""
Created on Wed May  1 13:51:20 2019

@author: jmiller
"""

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Run this line only if you want to print entire/untruncated outputs
pd.set_option('display.max_columns', None)

# Load dataset
boston = load_boston()

# Dataset basics
print(boston['DESCR'] + '\n' + '-' * 79)
print('\nDataset keys: \n' + str(list(boston)))
print('\nFeatures: \n{}'.format(boston['feature_names']))
print('\nSize: \n{}'.format(boston.data.shape))
print('\nMissing Values: \n{}\n'.format(np.isnan(boston.data).sum()))

# Create dataframe
df = pd.DataFrame(boston.data, columns = boston['feature_names'])
df['MEDV'] = boston.target
print(df.head())
print()
print(df.describe())

# Split into features/target and train/test sets
X = df[df.columns[:-1]]
y = df['MEDV']
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size = 0.2,
                                                    random_state = 42)

# TODO scale features

# Linear SVR - create, train and predict
linSVR_clf = SVR(kernel = 'linear')
linSVR_clf.fit(X_train, y_train)
linSVR_pred = linSVR_clf.predict(X_test)

# Check MSE
print('\nMean Squared Error: {}'.format(mean_squared_error(
                                        y_test, linSVR_pred)))

# Plot predicted vs. actual
plt.scatter(y_test, linSVR_pred, s = 10)
plt.plot(np.arange(0, 60, 10), np.arange(0, 60, 10), 'k-.',
         linewidth = 3,
         alpha = 0.7)
plt.title('Boston Housing Prices ($1000s) - SVR w/ Linear Kernel')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.xlim(-10, 60)
plt.ylim(-10, 60)
plt.show()

# RBF SVR
# Maybe use grid search on c and gamma here
rbfSVR_clf = SVR(kernel = 'rbf', gamma = 'scale', C = 100)
rbfSVR_clf.fit(X_train, y_train)
rbfSVR_pred = rbfSVR_clf.predict(X_test)

# Check MSE
print('\nMean Squared Error: {}'.format(mean_squared_error(
                                        y_test, rbfSVR_pred)))

# Plot predicted vs. actual
plt.scatter(y_test, rbfSVR_pred, s = 10)
plt.plot(np.arange(0, 60, 10), np.arange(0, 60, 10), 'k-.',
         linewidth = 3,
         alpha = 0.7)
plt.title('Boston Housing Prices ($1000s) - SVR w/ RBF Kernel')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.xlim(-10, 60)
plt.ylim(-10, 60)
plt.show()

# TODO if few features important: Lasso, ElasticNet
# TODO otherwise RidgeRegression, SVR(kernel = linear, rbf), EnsembleRegressors
# TODO try with cross validation