
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

np.random.seed(42)

features = [
    "attendance",
    "internal_marks",
    "assignment_marks",
    "previous_score"
]

# Generate sample student data
X = pd.DataFrame(
    np.random.randint(0, 101, size=(2000, 4)),
    columns=features
)

# 1 = PASS, 0 = FAIL
y = (X.mean(axis=1) >= 50).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = DecisionTreeClassifier(max_depth=6, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

joblib.dump(model, "student_result_model.pkl")

with open("metrics.json", "w") as f:
    json.dump({"accuracy": accuracy}, f, indent=4)

print("Model training completed.")
print("Model accuracy:", round(accuracy, 4))
