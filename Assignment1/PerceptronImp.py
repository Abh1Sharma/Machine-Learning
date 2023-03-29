import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import confusion_matrix 

def fit_perceptron(X_train, y_train):
    #Creates the first column of 1's in our matrix
    bias = np.ones((X_train.shape[0], 1))
    X_train = np.hstack((bias, X_train))
    #Intial weight vector is set to 0
    w = np.zeros(X_train.shape[1])
    epochs = 5000
    #Loops over the number of assigned epochs along with the rows of the matrix
    for t in range(epochs):
        for i, x in enumerate(X_train):
            #Dot product is checked to see whether the predicted value is equal to true value
            if (np.dot(x, w) * y_train[i]) <= 0:
                #Weight update rule is issued
                w = w + x * y_train[i]
    return w 

def errorPer(X_train,y_train,w):
    #Creates the first column of 1's in our matrix 
    bias = np.ones((X_train.shape[0], 1))
    X_train = np.hstack((bias,X_train))
    negCount = 0
    for i in range(len(X_train)):
        #Checks the number of instances where the predicted value is not equal to the true value
        if pred(X_train[i,:],w) != y_train[i]:
            negCount += 1
    avgError = negCount/len(X_train)
    return avgError

def confMatrix(X_train,y_train,w):
    predicted_y = pred(X_train, w)
    true_neg = 0
    false_pos = 0
    false_neg = 0
    true_pos = 0
    #Total is calculated for each of the 4 possibilites of outcomes
    true_neg = np.sum((predicted_y == -1) & (y_train == -1))
    false_pos = np.sum((predicted_y == 1) & (y_train == -1))
    false_neg = np.sum((predicted_y == -1) & (y_train == 1))
    true_pos = np.sum((predicted_y == 1) & (y_train == 1))
    #use of np.sum aggregates the values of confusion matrix based on prediction and train values
    return np.array([[true_neg, false_pos], [false_neg, true_pos]])

def pred(X_train, w):
    #Creates the first column of 1's in our matrix 
    ones = np.ones((X_train.shape[0], 1))
    X_train = np.hstack((ones, X_train))
    #Checks the dot product and returns a corresponding value depending on whether the result was positive or negative 
    predicted_y = np.dot(X_train, w)
    for i in range(predicted_y.shape[0]):
        if predicted_y[i] > 0:
            predicted_y[i] = 1
        elif predicted_y[i] <= 0:
            predicted_y[i] = -1
    return predicted_y

def test_SciKit(X_train, X_test, Y_train, Y_test):
    #Uses the buil-in Perceptron algorithm which updates the weights accoridnly via pc.fit
    pct=Perceptron()
    pct.fit(X_train,Y_train)
    #Makes prediction for the matrix and returns the corresponding confusion matrix
    pred_pct=pct.predict(X_test)
    return confusion_matrix(Y_test,pred_pct)

def test_Part1():
    from sklearn.datasets import load_iris
    X_train, y_train = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X_train[50:],y_train[50:],test_size=0.2)

    #Set the labels to +1 and -1
    y_train[y_train == 1] = 1
    y_train[y_train != 1] = -1
    y_test[y_test == 1] = 1
    y_test[y_test != 1] = -1

    #Pocket algorithm using Numpy
    w=fit_perceptron(X_train,y_train)
    cM=confMatrix(X_test,y_test,w)

    #Pocket algorithm using scikit-learn
    sciKit=test_SciKit(X_train, X_test, y_train, y_test)
    
    #Print the result
    print ('--------------Test Result-------------------')
    print("Confusion Matrix is from Part 1a is: ",cM)
    print("Confusion Matrix from Part 1b is:",sciKit)
    

test_Part1()
