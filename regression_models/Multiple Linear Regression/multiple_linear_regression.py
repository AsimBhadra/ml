#multiple linear regression

#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing datasets
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,4].values

#encoding categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

#avoiding the dummy variable trap
X = X[:,1:]

#splitting data into training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.2, random_state = 0)


"""#feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)""" 
 
#fitting multiple linear regression to training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)

#predicting test set results
Y_pred = regressor.predict(X_test)

#building the optimal model using Backward elimination
import statsmodels.formula.api as sm
#adding column of ones as statsmodel requires
X = np.append(arr = np.ones((50,1)).astype(int),values =X,axis =1)

#selecting all variables as all-in
import statsmodels.api as sm
X_opt = np.array(X[:, [0, 1, 2, 3, 4, 5]], dtype=float)
#OrdinaryLeastSquares
regressor_OLS = sm.OLS(endog=Y, exog=X_opt).fit()
print(regressor_OLS.summary())

X_opt = np.array(X[:, [0, 1, 3, 4, 5]], dtype=float)
regressor_OLS = sm.OLS(endog=Y, exog=X_opt).fit()
print(regressor_OLS.summary())

X_opt = np.array(X[:, [0, 3, 4, 5]], dtype=float)
regressor_OLS = sm.OLS(endog=Y, exog=X_opt).fit()
print(regressor_OLS.summary())

X_opt = np.array(X[:, [0, 3, 5]], dtype=float)
regressor_OLS = sm.OLS(endog=Y, exog=X_opt).fit()
print(regressor_OLS.summary())

X_opt = np.array(X[:, [0, 3]], dtype=float)
regressor_OLS = sm.OLS(endog=Y, exog=X_opt).fit()
print(regressor_OLS.summary())