#polynomial regression

#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing datasets
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:-1].values
Y = dataset.iloc[:,-1].values

'''#splitting data into training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.2, random_state = 0)'''

'''not splitting data into training and test set as we have very less data and we need very
accurate results'''

"""#feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)""" 

'''no need to feature scale as we are using library that 
feature scales automatically to produce accurate results'''

#comparing linear and polynomial regression

#fitting linear regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,Y)

#fitting polynomial regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
'''poly_reg transforms matrix of features x to new matrix of features x_poly  
with x1, x1**2, x1**3 upto x1**n as we want'''
poly_reg = PolynomialFeatures(degree = 8)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, Y)

#visualising linear regression results
plt.scatter(X,Y,color = 'red')
plt.plot(X,lin_reg.predict(X),color = 'blue')
plt.title('Real salary vs Prediction using Linear regression')
plt.xlabel('Position label')
plt.ylabel('Salary')
plt.show()

# Visualising the Polynomial Regression results
plt.scatter(X, Y, color = 'red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

#predicting new result with linear regression
print(lin_reg.predict([[6.5]]))

#predicting new result with polynomial regression
print(lin_reg_2.predict(poly_reg.fit_transform([[6.5]])))










