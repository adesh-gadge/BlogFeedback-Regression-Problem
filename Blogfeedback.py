# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 14:59:48 2019

@author: Adesh
"""
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt

### Reading the train data
df_train = pd.read_csv('blogData_train.csv',header= None )
df_train.head()
X_train = df_train.iloc[:,:280]
Y_train = df_train.iloc[:,280]
Y_train.head()
### Reading the test data
df_test = pd.read_csv('blogData_test-2012.03.31.01_00.csv',header= None )
X_test = df_test.iloc[:,:280]
Y_test = df_test.iloc[:,280]
X_test.head()

###### OLS linear regression#####
reg = linear_model.LinearRegression()
reg.fit(X_train,Y_train)
reg.coef_

##### training MSE ########
Y_pred_train = reg.predict(X_train)
print("Linear Regression train Root Mean squared error: %.2f"
      % np.sqrt(mean_squared_error(Y_train, Y_pred_train)))
###### testing MSE #####
Y_pred = reg.predict(X_test)
print("Linear Regression test Root Mean squared error: %.2f"
      % np.sqrt(mean_squared_error(Y_test, Y_pred)))

# ridge regression
reg_ridge = linear_model.Ridge(alpha=0.1)
reg_ridge.fit(X_train,Y_train)
reg_ridge.coef_

# training MSE
Y_pred_train = reg_ridge.predict(X_train)
print("Ridge Linear Regression train Root Mean squared error: %.2f"
      % np.sqrt(mean_squared_error(Y_train, Y_pred_train)))
# testing MSE
Y_pred_ridge = reg_ridge.predict(X_test)
print("Ridge Linear Regression test Root Mean squared error: %.2f"
      % np.sqrt(mean_squared_error(Y_test, Y_pred_ridge)))

# Cross validated ridge regression
reg_ridge_cv =linear_model.RidgeCV(alphas=np.logspace(-6, 6, 13))
reg_ridge_cv.fit(X_train,Y_train)
reg_ridge_cv.alpha_
print('Cross validated value of alpha for ridge regression : %.5f' % reg_ridge_cv.alpha_ )
Y_pred_train = reg_ridge_cv.predict(X_train)
# training MSE
print("Cross validated Ridge Linear Regression train Root Mean squared error: %.2f"
      % np.sqrt(mean_squared_error(Y_train, Y_pred_train)))
# testing MSE
Y_pred_ridge_cv = reg_ridge_cv.predict(X_test)
print("Cross validated Ridge Linear Regression test Root Mean squared testing error: %.2f"
      % np.sqrt(mean_squared_error(Y_test, Y_pred_ridge_cv)))


# Lasso regression
reg_lasso = linear_model.Lasso(alpha=0.1)
reg_lasso.fit(X_train,Y_train)
reg_lasso.coef_

# training MSE
Y_pred_train = reg_lasso.predict(X_train)
print("Lasso Linear Regression train Root Mean squared error: %.2f"
      % np.sqrt(mean_squared_error(Y_train, Y_pred_train)))
# testing MSE
Y_pred_ridge = reg_lasso.predict(X_test)
print("Lasso Linear Regression test Root Mean squared error: %.2f"
      % np.sqrt(mean_squared_error(Y_test, Y_pred_ridge)))

# Cross validated lasso regression
reg_lasso_cv =linear_model.LassoCV(alphas=np.logspace(-6, 6, 13))
reg_lasso_cv.fit(X_train,Y_train)
reg_lasso_cv.alpha_
print('Cross validated value of alpha for lasso regression : %.5f' % reg_lasso_cv.alpha_ )
Y_pred_train = reg_ridge_cv.predict(X_train)
# training MSE
print("Cross Validated Lasso Linear Regression train Root Mean squared error: %.2f"
      % np.sqrt(mean_squared_error(Y_train, Y_pred_train)))
# testting MSE
Y_pred_ridge_cv = reg_ridge_cv.predict(X_test)
print("Cross Validated  Lasso Linear Regression testing Root Mean squared  error: %.2f"
      % np.sqrt(mean_squared_error(Y_test, Y_pred_ridge_cv)))

# feature importance plot
plt.plot(reg_lasso_cv.coef_,alpha=reg_lasso_cv.alpha_,linestyle='none',marker='*',markersize=5,color='red',label=r'Lasso; $\alpha = 1$',zorder=7)
plt.xlabel('Coefficient Index',fontsize=16)
plt.ylabel('Coefficient Magnitude',fontsize=16)
#plt.legend(fontsize=13,loc=4) 

# important featuers sorted
features_importance_sorted =np.argsort(np.absolute(reg_lasso_cv.coef_))[::-1]
# top 10 features
print(features_importance_sorted[0:10])







