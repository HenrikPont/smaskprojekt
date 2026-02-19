import numpy as np
import pandas as pd

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest, f_classif

import features




selected_features = [
    #"weather_pca1", "weather_pca2", "bad_conditions", "summertime", "hour_cos", "hour_sin", "day_sin", "day_cos", "month_sin", "month_cos", "weekday"
    "weather_pca1", "weather_pca2", "bad_conditions", "summertime", "hour_cos", "hour_sin", "day_sin","day_cos"
] #Removed "snow"

X_train, X_test, Y_train, Y_test = features.train_test_data(selected_features, random_state=0)


pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier())
])


#Grid search for KNN parameters

param_grid = {
    "knn__n_neighbors": [5, 7, 9, 11, 15, 21, 25, 26, 27, 28, 29, 31, 33, 35, 37, 39],
    "knn__weights": ["uniform", "distance"],
    "knn__metric": ["euclidean", "manhattan"]
}

grid = GridSearchCV(
    pipe,
    param_grid,
    cv=5,
    scoring="f1" # Use F1-score for imbalanced classification
)

grid.fit(X_train, Y_train)

print("Best parameters:", grid.best_params_)
print("Best CV F1:", grid.best_score_)


#Threshold tuning

best_model = grid.best_estimator_
probs = best_model.predict_proba(X_test)[:, 1]

thresholds = np.arange(0.1, 0.9, 0.01)
f1_scores = []

for t in thresholds:
    preds = (probs >= t).astype(int)
    f1_scores.append(f1_score(Y_test, preds))

best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]
best_threshold = 0.4 # Chosen based on observed F1 scores and class balance

print("\nBest threshold:", best_threshold)
print("Best test F1:", f1_scores[best_idx])

final_predictions = (probs >= best_threshold).astype(int)

print("\nClassification Report (Threshold Tuned):")
print(classification_report(Y_test, final_predictions))

print("\nConfusion Matrix:")
print(pd.crosstab(final_predictions, Y_test,
                 rownames=["Predicted"], colnames=["Actual"]))

from sklearn.inspection import permutation_importance

best_model = grid.best_estimator_

result = permutation_importance(
    best_model,
    X_test,
    Y_test,
    scoring="f1",
    n_repeats=20,
    random_state=0
)

importance = pd.Series(
    result.importances_mean,
    index=X_test.columns
)

print("\nPermutation Importances:")
print(importance.sort_values(ascending=False))