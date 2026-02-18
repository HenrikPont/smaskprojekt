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

url = 'training_data_VT2026.csv'
features = pd.read_csv(url, na_values='?', dtype={'ID': str}).dropna()

def heat_index(T_C, RH): # Steadman heat index formula
    # Convert to Fahrenheit
    T_F = T_C * 9/5 + 32
    HI_F = (-42.379 + 2.04901523*T_F + 10.14333127*RH 
            - 0.22475541*T_F*RH - 0.00683783*T_F**2 
            - 0.05481717*RH**2 + 0.00122874*T_F**2*RH 
            + 0.00085282*T_F*RH**2 - 0.00000199*T_F**2*RH**2)
    # Convert back to Celsius
    return (HI_F - 32) * 5/9

def combined_features():
    url = 'training_data_VT2026.csv'
    features = pd.read_csv(url, na_values='?', dtype={'ID': str}).dropna()
    
    features["bad_conditions"] = np.where(
        (features["precip"] > 1) |
        (features["snowdepth"] > 0) |
        (features["windspeed"] > 30) |
        (features["visibility"] < 14),
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

    #features["heat_index"] = heat_index(features["temp"], features["humidity"])

    return features