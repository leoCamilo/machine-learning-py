# Simple Linear Regression

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv('')
dataset = pd.read_csv('../data/Salary_Data.csv')

#think as a function, X is the variable, y is the answer for that variable (dependent variable)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)
x_lr_pred = regressor.predict(X_train)	#lr = linear regression

plt.scatter(X_train, y_train, color = 'red')
plt.scatter(X_test, y_test, color = 'green')
plt.plot(X_train, x_lr_pred, color = 'blue')
plt.title('Salary vs Experience (red - training set | green - test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()