import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

dataset = pd.read_csv('BIKE DETAILS.csv')

dataset['current_year'] = 2021
dataset['diff_year'] = dataset['current_year'] - dataset['year']
dataset.drop(['year', 'current_year', 'name'], axis=1, inplace=True)
dataset.replace({'1st owner':1, '2nd owner':2, '3rd owner':3, '4th owner':4}, inplace=True)

dataset = pd.get_dummies(dataset, drop_first=True)
dataset.dropna(inplace=True)

X = dataset.iloc[:, 1:].values
y = dataset.iloc[:, 0].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

pickle.dump(regressor, open('model.pkl', 'wb'))
model = pickle.load(open('model.pkl', 'rb'))
