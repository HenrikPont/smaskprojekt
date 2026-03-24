import numpy as np
import pandas as pd

from imblearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA

from features import combined_features

import csv

#Train model -----------------------------------------------------------------------------

selected_features = [
    "weather_pca1", "weather_pca2", "bad_conditions", "summertime", "hour_cos", "hour_sin", "day_sin","day_cos"
]

df = combined_features()[selected_features + ["increase_stock"]].dropna()
X = df.drop(columns=["increase_stock"])
y = (df["increase_stock"] == "high_bike_demand").astype(int)

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=25, weights="uniform", metric="manhattan"))
])

model = pipe.fit(X, y)


#Predict on test data -----------------------------------------------------------------------

url = 'test_data_VT2026.csv'
X_test = pd.read_csv(url, na_values='?', dtype={'ID': str}).dropna()

# Create 'bad_conditions' feature based on weather conditions
X_test["bad_conditions"] = np.where(
    (X_test["precip"] > 1) |
    (X_test["snowdepth"] > 0) |
    (X_test["windspeed"] > 32) |
    (X_test["visibility"] < 15.9),
    1,  # bad condition
    0   # good condition
)

# Perform PCA on weather-related features
pca_features = X_test[["temp", "dew", "humidity"]]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(pca_features)
pca = PCA(n_components=3)
components = pca.fit_transform(X_scaled)
X_test["weather_pca1"] = components[:, 0]
X_test["weather_pca2"] = components[:, 1]
X_test["weather_pca3"] = components[:, 2]

# Create cyclical features for hour, month, and day of week
X_test['hour_sin'] = np.sin(2*np.pi*X_test['hour_of_day'] /24)
X_test['hour_cos'] = np.cos(2*np.pi*X_test['hour_of_day'] /24)

X_test['month_sin'] = np.sin(2*np.pi*(X_test['month']) /12)
X_test['month_cos'] = np.cos(2*np.pi*(X_test['month']) /12)

X_test['day_sin'] = np.sin(2*np.pi*(X_test['day_of_week']) /7)
X_test['day_cos'] = np.cos(2*np.pi*(X_test['day_of_week']) /7)


# Results
probs = model.predict_proba(X_test[selected_features])[:, 1]

predictions = (probs >= 0.4).astype(int)

print("Predictions:", predictions)
print(len(predictions))

print(probs)

with open('predictions.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(predictions)