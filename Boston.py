# -*- coding: utf-8 -*-
"""
Created on Wed May  1 13:51:20 2019

@author: jmiller
"""

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
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

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)



def plot_compare(classifier):
    '''
    Plots predicted results vs. actual results
    '''
    predicted = classifier.predict(X_test)
    mse = round(mean_squared_error(y_test, predicted), 2)
    plt.scatter(y_test, predicted, s = 10)
    plt.plot(np.arange(0, 60, 10), np.arange(0, 60, 10), 'k-.',
             linewidth = 3,
             alpha = 0.7)
    try:
        plt.title('SVR w/ {} Kernel, MSE: {}'.format(
                  classifier.get_params()['kernel'].upper(),
                  mse))
    except KeyError:
        plt.title('Grid Search - C: {}, Kernel: {} - MSE: {}'.format(
                grid_clf.best_params_['C'],
                grid_clf.best_params_['kernel'].upper(),
                mse))
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.xlim(-10, 60)
    plt.ylim(-10, 60)
    plt.show()

# Linear SVR - create, train and predict
linSVR_clf = SVR(kernel = 'linear')
linSVR_clf.fit(X_train, y_train)
linSVR_pred = linSVR_clf.predict(X_test)
plot_compare(linSVR_clf)

# RBF SVR
rbfSVR_clf = SVR(kernel = 'rbf', gamma = 'scale', C = 100)
rbfSVR_clf.fit(X_train, y_train)
rbfSVR_pred = rbfSVR_clf.predict(X_test)
plot_compare(rbfSVR_clf)

print()

# GridSearch - MUCH faster with scaling features
parameters = {'kernel': ('linear', 'rbf'),
              'C': [0.01, 0.1, 1, 10, 100, 1000]}
grid_svr = SVR(gamma = 'scale')
grid_clf = GridSearchCV(grid_svr,
                        parameters,
                        cv = 5,
                        scoring = 'neg_mean_squared_error',
                        verbose = 1)
grid_clf.fit(X_train, y_train)
grid_pred = grid_clf.predict(X_test)

# Get some useful results from GridSearch
print('\n--- Grid Search Results ---')
print('\nBest Estimator:\n' + str(grid_clf.best_estimator_))
print('\nScore:\n' + str(grid_clf.best_score_) + '\n')

plot_compare(grid_clf)

# TODO if few features important: Lasso, ElasticNet
# TODO otherwise RidgeRegression, SVR(kernel = linear, rbf), EnsembleRegressors