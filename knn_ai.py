import numpy as np
import pandas as pd

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold, SequentialFeatureSelector
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

# Load data
url = 'training_data_VT2026.csv'
bikes = pd.read_csv(url, na_values='?', dtype={'ID': str}).dropna()

np.random.seed(0)
trainI = np.random.choice(bikes.shape[0], size=800, replace=False)
trainIndex = bikes.index.isin(trainI)
train = bikes.iloc[trainIndex]
test = bikes.iloc[~trainIndex]

# Features
all_features = [
    "hour_of_day","day_of_week","month",
    "holiday","weekday","summertime",
    "temp","dew","humidity",
    "precip","snowdepth",
    "windspeed","cloudcover","visibility"
]

weather_features = ["temp","dew","humidity"]
other_features = [f for f in all_features if f not in weather_features]

# Prepare X and Y
X_train = train[all_features].copy()
Y_train = np.where(train['increase_stock'] == 'high_bike_demand', 1, 0)
X_test = test[all_features].copy()
Y_test = np.where(test['increase_stock'] == 'high_bike_demand', 1, 0)

# ColumnTransformer: PCA for weather features, scale others
preprocessor = ColumnTransformer([
    ("weather_pca", PCA(n_components=2), weather_features),
    ("scale_others", StandardScaler(), other_features)
], remainder='passthrough')

# Base KNN classifier
knn = KNeighborsClassifier()

# Full pipeline with backward SequentialFeatureSelector
pipe = Pipeline([
    ("preprocess", preprocessor),               # PCA + scaling
    ("smote", SMOTE(random_state=42, k_neighbors=3)),  # balance classes
    ("var", VarianceThreshold()),               # remove constant features
    ("sfs", SequentialFeatureSelector(
        knn,
        n_features_to_select=8,                 # choose desired number of final features
        direction='backward',                   # backward selection
        scoring='f1',
        cv=5
    )),
    ("knn", knn)                                # final KNN classifier
])

# Hyperparameter grid for the final KNN
param_grid = {
    "knn__n_neighbors": [5, 7, 9, 11, 15],
    "knn__weights": ["uniform", "distance"],
    "knn__metric": ["euclidean", "manhattan"]
}

# Grid search
grid = GridSearchCV(pipe, param_grid, cv=5, scoring='f1')
grid.fit(X_train, Y_train)

# Evaluate on test set
predictions = grid.predict(X_test)
print("Best parameters:", grid.best_params_)
print("Best CV F1:", grid.best_score_)
print("\nClassification report:")
print(classification_report(Y_test, predictions))
print('Confusion matrix:\n')
print(pd.crosstab(predictions, Y_test, rownames=['Predicted'], colnames=['Actual']))

# Show selected features after backward SFS
best_pipe = grid.best_estimator_
X_train_transformed = best_pipe.named_steps['preprocess'].transform(X_train)
sfs_support = best_pipe.named_steps['sfs'].get_support()
# Feature names after preprocessing: PCA components + scaled other features
pc_names = [f"PC{i+1}" for i in range(2)]
all_transformed_features = pc_names + other_features
selected_features_names = np.array(all_transformed_features)[sfs_support]
print("Selected features after backward SFS:", selected_features_names.tolist())
