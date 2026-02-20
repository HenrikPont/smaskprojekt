import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sklearn.preprocessing as skl_pre
import sklearn.linear_model as skl_lm
import sklearn.discriminant_analysis as skl_da
import sklearn.neighbors as skl_nb
from sklearn import model_selection as ms
from sklearn.metrics import classification_report

import features 

#Hämta faeatures
bikes = features.combined_features()

selected_features = [
    'weather_pca1', 'weather_pca2', 
    'hour_sin', 'hour_cos',
    'day_sin', 'day_cos',
    'month_sin', 'month_cos',
    'bad_conditions',
    'holiday',
    'windspeed',
]

X = bikes[selected_features]
y = bikes['increase_stock']

np.random.seed(0)

trainI = np.random.choice(bikes.shape[0], size=800, replace=False)
trainIndex = bikes.index.isin(trainI)
train = bikes.iloc[trainIndex]
test = bikes.iloc[~trainIndex]

X_train = train[selected_features]
y_train = np.where(train['increase_stock'] == 'high_bike_demand', 1, 0)
X_test = test[selected_features]
y_test = np.where(test['increase_stock'] == 'high_bike_demand', 1, 0)

# 3. Splitta datan
#X_train, X_test, y_train, y_test = ms.train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Skapa och träna QDA
# Tips: Vi kan manuellt sätta "priors" om vi vill tvinga modellen 
# att vara mer vaksam på "high_bike_demand"
model = skl_da.QuadraticDiscriminantAnalysis()
model.fit(X_train, y_train)

# 5. Utvärdera
prediction = model.predict(X_test)

print("Confusion Matrix:")
print(pd.crosstab(prediction, y_test, rownames=['Predicted'], colnames=['Actual']))
print("\nClassification Report:")
print(classification_report(y_test, prediction))







