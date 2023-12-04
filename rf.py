import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

data = pd.read_csv("angle.csv")
X = data.iloc[:,:-1]
y = data.iloc[:,4]
print(y)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size =0.3, random_state=1)
classifier = RandomForestClassifier(n_estimators=100, criterion='log_loss', random_state=42)
classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)
print(accuracy_score(y_test,y_pred))