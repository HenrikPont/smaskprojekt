import numpy as np
import pandas as pd

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest, f_classif

url = 'training_data_VT2026.csv'
bikes = pd.read_csv(url, na_values='?', dtype={'ID': str}).dropna()

bikes["bad_conditions"] = np.where(
    (bikes["precip"] > 1) |
    (bikes["snowdepth"] > 0) |
    (bikes["windspeed"] > 30) |
    (bikes["visibility"] < 14),
    1,  # bad condition
    0   # good condition
)

np.random.seed(0)

trainI = np.random.choice(bikes.shape[0], size=800, replace=False)
trainIndex = bikes.index.isin(trainI)
train = bikes.iloc[trainIndex]
test = bikes.iloc[~trainIndex]



features = [
    "hour_of_day","day_of_week","month",
    "holiday","weekday","summertime",
    "temp","dew","humidity",
    "bad_conditions","cloudcover"
] #Removed "snow"

X_train = train[features]
Y_train = np.where(train['increase_stock'] == 'high_bike_demand', 1, 0)
X_test = test[features]
Y_test = np.where(test['increase_stock'] == 'high_bike_demand', 1, 0)


pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("smote", SMOTE(random_state=0)),
    ("select", SelectKBest(score_func=f_classif)),
    ("knn", KNeighborsClassifier())
])

param_grid = {
    "select__k": [4, 5, 6, 7, 8, 9, 10, 12, 14],
    "knn__n_neighbors": [5, 7, 9, 10, 11, 13, 15, 21, 31],
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

predictions = grid.predict(X_test)
print(classification_report(Y_test, predictions))
print('Confusion matrix:\n')
print(pd.crosstab(predictions, Y_test, rownames=['Predicted'], colnames=['Actual']))

# Access the fitted pipeline that gave the best CV score
best_pipe = grid.best_estimator_

# Get the features selected by SelectKBest inside the fitted pipeline
selected_features = X_train.columns[best_pipe.named_steps['select'].get_support()]
print("Selected features:", selected_features.tolist())