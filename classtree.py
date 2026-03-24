import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn import tree

import features 

#Get features
bikes = features.combined_features()

#selected_features = ['hour_of_day', 'day_of_week', 'month', 'temp', 'dew', 'humidity', 'precip', 'snowdepth', 'windspeed', 'visibility']

selected_features = [
    'weather_pca1', 'weather_pca2', 
    'hour_sin', 'hour_cos',
    'day_sin', 'day_cos',
    'month_sin', 'month_cos',
    'bad_conditions',
    'summertime',
]

X_train, X_test, y_train, y_test = features.train_test_data(selected_features, random_state=0)

depth = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
highest_acc = 0
max_d = 1
for i in depth:
    model = tree.DecisionTreeClassifier(max_depth=i)
    model.fit(X=X_train, y=y_train)
    y_predict = model.predict(X_test)
    acc = accuracy_score(y_test, y_predict)
    if acc > highest_acc:
        highest_acc = acc
        max_d = i

print(f'max depth = {max_d}')

model = tree.DecisionTreeClassifier(max_depth=max_d)
model.fit(X=X_train, y=y_train)
y_predict = model.predict(X_test)

#Check importance of variables
importance_Bike = pd.DataFrame({"Feature": X_train.columns,"Importance": model.feature_importances_}).sort_values(by="Importance", ascending=False)
print(importance_Bike)


print(classification_report(y_test, y_predict))