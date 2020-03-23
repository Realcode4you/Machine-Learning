#import libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

#Assign Index Name

column_names = ["CRIM","ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX","PTRATIO","B","LSTAT","MEDV"]

#Read US housing data

datafile = "housing.data"
dataFrame = pd.read_csv(datafile,header=None, delim_whitespace = True, names = column_names)

prices = dataFrame['MEDV']
features = dataFrame.drop('MEDV', axis = 1)

mean = dataFrame['MEDV'].mean()
mean

median = np.median(dataFrame['MEDV'])
median

prices.head()

features.head()

print('Boston housing dataset has {0} data points with {1} variables each'.format(*dataFrame.shape))

#Split Main and target(price) data


# TODO: Shuffle and split the data into training and testing subsets
X_train, X_test, y_train, y_test = train_test_split(features, prices, test_size=0.2, random_state=10)


X_train.shape

X_test.shape

#Implement a linear regression model with ridge regression that predicts median house prices from the other variables.

# initialize
from sklearn.linear_model import Ridge
from sklearn import metrics


## training the model

ridgeReg = Ridge(alpha=0.05, normalize=True)

ridgeReg.fit(X_train,y_train)

pred_X = ridgeReg.predict(X_test)

pred_X

ridgeReg.score(X_test,y_test)

### Use 10-fold cross validation on 80-20 train-test splits and report the final R2 values that you discovered. 

n = 10  # repeat the CV procedure 10 times to get more precise results

for i in range(n):
    # for each iteration, randomly hold out 20% of the data as CV set
    X_train1, X_cv, y_train1, y_cv = train_test_split(
    X_train, y_train, test_size=.20, random_state=42)

    # train model and make predictions
    ridgeReg.fit(X_train1, y_train1) 
    pred_X = ridgeReg.predict(X_cv)
    
#predicted vaalue    
pred_X

ridgeReg.score(X_cv,y_cv)

#predicting X_train1
predict_train = ridgeReg.predict(X_train1)


@R2 Score
from sklearn.metrics import r2_score
def performance_metric(y_train1, predict_train):
    score = r2_score(y_train1, predict_train)

    # Return the score
    return score

performance_metric(y_train1, predict_train)


