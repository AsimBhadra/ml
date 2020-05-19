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

#fitting regression model to the dataset

#regressor here


#predicting new result with regression model
y_pred = regressor.predict([[6.5]])
# Visualising the Polynomial Regression results
plt.scatter(X, Y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Regression Model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualising the Polynomial Regression results with higher res. and smoother curve
X_grid = np.arange(min(X),max(X),0.1)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X, Y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Regression Model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()










