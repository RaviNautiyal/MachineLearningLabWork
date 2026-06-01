Implementing the simple linear regression with custom class


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
class CustomLinearRegresssion:
    def __init__(self):
        self.m = None
        self.b = None
    def fit(self,X_train,y_train):
      num = 0
      den = 0
      for i in range(X_train.shape[0]):
        num = num + ((X_train[i] - X_train.mean())*(y_train[i] - y_train.mean()))
        den = den + ((X_train[i] - X_train.mean())*(X_train[i] - X_train.mean()))

      self.m = num/den
      self.b = y_train.mean() - (self.m * X_train.mean())
    def predict(self,X_test):
      return self.m * X_test + self.b



pd.read_csv("placements.csv")
df = pd.DataFrame(pd.read_csv("placements.csv"))

X = df.iloc[:, 0].values
y= df.iloc[:, 1].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2 , random_state=2)

lr = CustomLinearRegresssion()


lr.fit(X_train,y_train)

print(X_test,y_test)
print(lr.predict(X_test))
**Implementing the custom Multiple Linear Regression with custom class**
from sklearn.datasets import load_diabetes

X, y = load_diabetes(return_X_y= True)
X.shape
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)
model = LinearRegression()
model.fit(X_train,y_train)
model.predict(X_test)
from sklearn.metrics import r2_score
r2_score(y_test,model.predict(X_test))
model.coef_
model.intercept_
Now making our own multiple linear regression class

class CustomMultipleLinearRegression:
  def __init__(self):
    self.coef_ = None
    self.intercept_ = None
  def fit(self,X_train,y_train):
    X_train = np.insert(X_train,0,1,axis=1)
    betas = np.linalg.inv(np.dot(X_train.T,X_train)).dot(X_train.T).dot(y_train)
    self.intercept_ = betas[0]
    self.coef_ = betas[1:]
  def predict(self,X_test):
    y_pred = np.dot(X_test,self.coef_) + self.intercept_
    return y_pred
lr = CustomMultipleLinearRegression()
lr.fit(X_train, y_train)
prediction = lr.predict(X_test)
print(prediction)
