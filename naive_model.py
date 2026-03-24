import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import classification_report

import features

# --- Load and prepare data ---
X_train, X_test, Y_train, Y_test = features.train_test_data([], random_state=0)

final_predictions = np.zeros_like(Y_test)

# --- Results ---
print("\nClassification Report (Threshold Tuned):")
print(classification_report(Y_test, final_predictions))

print("\nConfusion Matrix:")
print(pd.crosstab(final_predictions, Y_test,
                 rownames=["Predicted"], colnames=["Actual"]))
