### FILL WITH PROPER FUNC

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import sklearn.datasets as datasets
from sklearn.preprocessing import PolynomialFeatures
# Cross-Validation tools
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
#  From Tut on Linear Regression ipynb


def fit_LinRegr(X_train, y_train):
  #Add implementation here
  arrOfOnes = np.ones((X_train.shape[0], 1))
  # X0 bias coordinate added to array
  newX = np.hstack((X_train, arrOfOnes))
  # pseudo inv calc: XT = inv(Xt*X)*XT
  XtX = np.dot(np.transpose(newX), newX)
  # (Xt*X)
  # print("pre inv:, ", XtX)
  invXtX = np.linalg.pinv(XtX)
  # print("inverted:, ", invXtX)
  # inv(Xt*X)
  XTfinal = np.dot(invXtX, np.transpose(newX))
  # inv(Xt*X)*Xt
  w = np.dot(XTfinal,y_train)
  # w = (XT)*y
  return w

def mse(X_train, y_train, w):
  hypothesis = pred(X_train, w)
  #  prediction vector to be tested against labelled vector y for Mean square error
  mse = np.mean((y_train - hypothesis)**2)
  return mse

def pred(X_train, w):
  # created array of ones thats the size of n rows in a nxm matrix. From X_train's rows size/shape[0]
  arrOfOnes = np.ones((X_train.shape[0], 1))
  # stacking/horizontal addition of the array of ones onto the X_train array
  newX = np.hstack((X_train, arrOfOnes))
  #  predicted weight is the dot product of currently trained X and input weight 
  newW = np.dot(newX, w)
  return newW  

def test_SciKit(X_train, X_test, Y_train, Y_test):
    # Load Library for linear model
    linearRegModel = linear_model.LinearRegression()
    #fit data of train set - to be tested against prediction set
    linearRegModel.fit(X_train, Y_train) #train model
    #  predictions on the test set, so using X_test. If wanted - use X_train and Y_train for MSE calc.
    predModel_test = linearRegModel.predict(X_test)
    mse = mean_squared_error(predModel_test, Y_test)

    return mse

def subtestFn():
    # This function tests if your solution is robust against singular matrix

    # X_train has two perfectly correlated features
    X_train = np.asarray([[1, 2], [2, 4], [3, 6], [4, 8]])
    y_train = np.asarray([1,2,3,4])
    
    try:
      w=fit_LinRegr(X_train, y_train)
      print ("weights: ", w)
      print ("NO ERROR")
    except:
      print ("ERROR")

def testFn_Part2():
    X_train, y_train = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X_train,y_train,test_size=0.2)
    
    w=fit_LinRegr(X_train, y_train)
    
    #Testing Part 2a
    e=mse(X_test,y_test,w)
    
    #Testing Part 2b
    scikit=test_SciKit(X_train, X_test, y_train, y_test)
    
    print("Mean squared error from Part 2a is ", e)
    print("Mean squared error from Part 2b is ", scikit)

print ('------------------subtestFn----------------------')
subtestFn()

print ('------------------testFn_Part2-------------------')
testFn_Part2()


'''
The Scikit value and my Function implementations are almost exact, wherein they are similar up to the 10th decimal point for example in the output we have:
Mean squared error from Part 2a is  3267.0862926789473
Mean squared error from Part 2b is  3267.0862926789796
'''