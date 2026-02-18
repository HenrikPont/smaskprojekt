import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sklearn.preprocessing as skl_pre
import sklearn.linear_model as skl_lm
import sklearn.discriminant_analysis as skl_da
import sklearn.neighbors as skl_nb
from sklearn import model_selection as ms
from sklearn.metrics import classification_report

from features import combined_features

#Hämta faeatures
df = combined_features()

selected_features = [
    'weather_pca1', 'weather_pca2', 
    'hour_sin', 'hour_cos',
    'day_sin', 'day_cos',
    'month_sin', 'month_cos',
    'bad_conditions',
    'holiday',
    'windspeed',
]

X = df[selected_features]
y = df['increase_stock']

# 3. Splitta datan
X_train, X_test, y_train, y_test = ms.train_test_split(X, y, test_size=0.2, random_state=42)

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







