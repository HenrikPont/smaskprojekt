import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sklearn.preprocessing as skl_pre
import sklearn.linear_model as skl_lm
import sklearn.discriminant_analysis as skl_da
import sklearn.neighbors as skl_nb
from sklearn import model_selection as ms
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

import features 

#Hämta faeatures
bikes = features.combined_features()

selected_features = [
    'weather_pca1', 'weather_pca2', 
    'hour_sin', 'hour_cos',
    'day_sin', 'day_cos',
    'month_sin', 'month_cos',
    'bad_conditions',
]



X_train, X_test, y_train, y_test = features.train_test_data(selected_features, random_state=0)

#Gridsearch 
reg_params = np.linspace(0, 0.5, 100) 
priors = [[p, 1-p] for p in np.linspace(0.1, 0.9, 9)] + [None]  

params = {
    'reg_param': reg_params,
    'priors': priors 
}

qda = skl_da.QuadraticDiscriminantAnalysis()

grid = GridSearchCV(qda, params, cv=5, scoring='f1')
grid.fit(X_train, y_train)

print(f"Bästa parametrar: {grid.best_params_}")
print(f"Bästa CV F1-score: {grid.best_score_:.4f}")
best_model = grid.best_estimator_

#Det man kom fram till
prediction = best_model.predict(X_test)

print("Confusion Matrix:")
print(pd.crosstab(prediction, y_test, rownames=['Predicted'], colnames=['Actual']))
print("\nClassification Report:")
print(classification_report(y_test, prediction))







