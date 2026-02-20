import numpy as np
import pandas as pd

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split


def combined_features():
    url = 'training_data_VT2026.csv'
    features = pd.read_csv(url, na_values='?', dtype={'ID': str}).dropna()
    
    features["bad_conditions"] = np.where(
        (features["precip"] > 1) |
        (features["snowdepth"] > 0) |
        (features["windspeed"] > 32) |
        (features["visibility"] < 15.9),
        1,  # bad condition
        0   # good condition
    )


    pca_features = features[["temp", "dew", "humidity"]]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(pca_features)
    pca = PCA(n_components=3)
    components = pca.fit_transform(X_scaled)
    features["weather_pca1"] = components[:, 0]
    features["weather_pca2"] = components[:, 1]
    features["weather_pca3"] = components[:, 2]
    print("Explained variance ratio:", pca.explained_variance_ratio_)

    features['hour_sin'] = np.sin(2*np.pi*features['hour_of_day'] /24)
    features['hour_cos'] = np.cos(2*np.pi*features['hour_of_day'] /24)

    features['month_sin'] = np.sin(2*np.pi*(features['month']-1) /12)
    features['month_cos'] = np.cos(2*np.pi*(features['month']-1) /12)

    features['day_sin'] = np.sin(2*np.pi*(features['day_of_week']) /7)
    features['day_cos'] = np.cos(2*np.pi*(features['day_of_week']) /7)

    #features["heat_index"] = heat_index(features["temp"], features["humidity"])

    return features


def train_test_data(selected_features, test_size=0.25, random_state=0):
    df = combined_features()[selected_features + ["increase_stock"]].dropna()
    X = df.drop(columns=["increase_stock"])
    y = (df["increase_stock"] == "high_bike_demand").astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    return X_train, X_test, y_train, y_test