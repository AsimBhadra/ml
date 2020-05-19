#simple linear regression
#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing datasets
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,1].values

 #splitting data into training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 1/3, random_state = 0)

#fitting simple linear regression to training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

#predicting the test set results
y_pred = regressor.predict(X_test)

#visualising the training set results
plt.scatter(X_train,Y_train,
            color = 'red')
plt.plot(X_train, regressor.predict(X_train),
         color = 'blue')
plt.title('Salary vs Experience(Training Set)')
plt.xlabel('Years of exp')
plt.ylabel('Salary')
plt.show()

#visualising the test set results
plt.scatter(X_test,Y_test,
            color = 'red')
plt.plot(X_train, regressor.predict(X_train),
         color = 'blue')
plt.title('Salary vs Experience(Test  Set)')
plt.xlabel('Years of exp')
plt.ylabel('Salary')
plt.show()










