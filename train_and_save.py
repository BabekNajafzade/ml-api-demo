import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib

data = {
    "age": [22, 25, 47, 52, 46, 56, 55, 60],
    "salary": [200, 250, 800, 900, 850, 1000, 950, 1100],
    "bought": [0, 0, 1, 1, 1, 1, 1, 1]
}

df = pd.DataFrame(data)

X = df[["age", "salary"]]
y = df["bought"]

model = LogisticRegression()
model.fit(X, y)

artifact = {
    "model": model,
    "feature_columns": X.columns.tolist(),
    "label_map": {0: "No", 1: "Yes"}
}

joblib.dump(artifact, "model.pkl")

print("model.pkl saved")
