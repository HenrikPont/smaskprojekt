import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn import tree


np.random.seed(0)
Bike = pd.read_csv('training_data_VT2026.csv')

#Drop variables with 0 importance
Bike

#remake hour of day and month to cyclic variables
Bike['hour_sin'] = np.sin(2*np.pi*Bike['hour_of_day'] /24)
Bike['hour_cos'] = np.cos(2*np.pi*Bike['hour_of_day'] /24)

Bike['month_sin'] = np.sin(2*np.pi*(Bike['month']-1) /12)
Bike['month_cos'] = np.cos(2*np.pi*(Bike['month']-1) /12)

Bike = Bike.drop(columns=['hour_of_day', 'month'])

#Combine dew and temp since strongly correlated
Bike['dew_temp'] = Bike['dew']+Bike['temp']
Bike = Bike.drop(columns=['dew', 'temp'])

#Sample
trainIndex = np.random.choice(Bike.shape[0], size=800, replace=False)
train = Bike.iloc[trainIndex]
test = Bike.iloc[~Bike.index.isin(trainIndex)]

#Train model
X_train = train.drop(columns=['increase_stock'])
y_train = train['increase_stock']

#Make classification tree
model = tree.DecisionTreeClassifier(max_depth=5)
model.fit(X=X_train, y=y_train)

#Test model
X_test = test.drop(columns=['increase_stock'])
y_test = test['increase_stock']
y_predict = model.predict(X_test)

#Check importance of variables
importance_Bike = pd.DataFrame({"Feature": X_train.columns,"Importance": model.feature_importances_}).sort_values(by="Importance", ascending=False)
print(importance_Bike)

#pd.plotting.scatter_matrix(Bike, figsize=(16,16))
#plt.show()

print(classification_report(y_test, y_predict))