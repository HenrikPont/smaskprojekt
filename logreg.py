import numpy as np
import sklearn.linear_model as skl_lm
import sklearn.metrics as skl_mt
import features

selected_features = ['holiday', 'weekday', 'summertime', 'bad_conditions', 'hour_sin', 'hour_cos', 'weather_pca2', 'month_sin', 'month_cos', 'day_sin', 'day_cos']

[X_train, X_test, y_train, y_test] = features.train_test_data(selected_features=selected_features)

model = skl_lm.LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)

y_predict = np.where(model.predict_proba(X_test)[:, 0] > 0.61, 0, 1) #thresholden blir baklänges av ngn anledning ??
print(skl_mt.classification_report(y_test, y_predict))