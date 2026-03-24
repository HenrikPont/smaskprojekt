import pandas as pd
import numpy as np
import sklearn.discriminant_analysis as skl_da
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GridSearchCV

import features 

# --- Load and prepare data ---
bikes = features.combined_features()

selected_features = [
    'weather_pca1', 'weather_pca2', 
    'hour_sin', 'hour_cos',
    'day_sin', 'day_cos',
    'month_sin', 'month_cos',
    'bad_conditions',
    'summertime',
]

X_train, X_test, y_train, y_test = features.train_test_data(selected_features, random_state=2)

# --- Gridsearch ---
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

# --- Threshold ---
probabilities = best_model.predict_proba(X_test)[:, 1]
thresholds = np.arange(0.1, 0.9, 0.01)
f1_scores = []

for threshold in thresholds:
    predictions = (probabilities >= threshold).astype(int)
    f1 = f1_score(y_test, predictions)
    f1_scores.append(f1)

best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]
print(f"Bästa tröskel: {best_threshold:.2f} med F1-score: {f1_scores[best_idx]:.4f}")


# --- Results ---
prediction = (probabilities >= best_threshold).astype(int)

print("Confusion Matrix:")
print(pd.crosstab(prediction, y_test, rownames=['Predicted'], colnames=['Actual']))
print("\nClassification Report:")
print(classification_report(y_test, prediction))







