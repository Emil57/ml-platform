import os

import joblib
from sklearn.linear_model import LinearRegression

from src.examples.ca_house_prediction.prepare_data import X_train, y_train

output_dir = os.path.join(
    os.path.abspath("../../"), "artifacts", "ca_house_prediction", "models"
)

try:
    print(f"Data Shape: {X_train.shape}, {y_train.shape}")
    model = LinearRegression()
    model.fit(X_train, y_train)

    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(model, f"{output_dir}/model.pkl")
except Exception as e:
    print(f"Error accessing training data: {e}")
