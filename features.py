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
features = pd.read_csv(url, na_values='?', dtype={'ID': str}).dropna()




features["bad_conditions"] = np.where(
    (features["precip"] > 1) |
    (features["snowdepth"] > 0) |
    (features["windspeed"] > 30) |
    (features["visibility"] < 14),
    1,  # bad condition
    0   # good condition
)

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

    

    return features