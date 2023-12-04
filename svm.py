import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score


def angle(co_1,co_2):
  cosine_angle = np.dot(co_1, co_2) / (np.linalg.norm(co_1) * np.linalg.norm(co_2))
  if cosine_angle<-1:
    cosine_angle=-1
  elif cosine_angle>1:
    cosine_angle=1
  angle = np.arccos(cosine_angle)
  return angle

file_origin="angle.csv"
Sports=pd.read_csv(file_origin)
X = Sports.iloc[:,:-1]
y = Sports.iloc[:,4]
X_train, X_test, Y_train, Y_test=train_test_split(X,y,test_size=0.3, random_state=1)
svm_linear=svm.SVC(C=100, kernel="rbf",decision_function_shape="ovr")
svm_linear.fit(X_train,Y_train)
print("train accuracy"+str(accuracy_score(Y_train,svm_linear.predict(X_train))))
print("test accuracy"+str(accuracy_score(Y_test,svm_linear.predict(X_test))))