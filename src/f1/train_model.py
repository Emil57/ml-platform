import os

import joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import SGDClassifier

from f1 import MODELS_DIR
from f1.feature_engineering import train, train_te, valid, valid_te

try:
    clf = SGDClassifier(
        loss="log_loss",  # logistic regression
        max_iter=1000,  # number of iterations
        class_weight={0: 1.0, 1: 20.0},  # handle imbalance
        random_state=42,
    )
    clf.fit(train_te, train["won"])

    # 11) Probability calibration (Platt scaling on validation)
    calibrated = CalibratedClassifierCV(estimator=clf, method="sigmoid", cv="prefit")
    calibrated.fit(valid_te, valid["won"])

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(calibrated, f"{MODELS_DIR}/model.pkl")
    print(f"Saved model to {MODELS_DIR}")
except Exception as e:
    print(f"Error in feature engineering: {e}")
    raise
