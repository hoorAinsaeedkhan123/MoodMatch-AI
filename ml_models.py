"""Training helpers for MoodMatch AI.

The included CSV is synthetic demo data created for learning purposes.
Logistic Regression predicts a fun entertainment-vibe category.
Linear Regression estimates a playful 0-100 vibe-energy score.
"""
from pathlib import Path
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = [
    "music_energy",
    "social_style",
    "movie_intensity",
    "creativity",
    "weekend_pace",
    "mystery_interest",
]
TARGET_CLASS = "vibe"
TARGET_SCORE = "vibe_energy_score"
DATA_PATH = Path(__file__).parent / "data" / "vibe_training_data.csv"


def load_data():
    """Load the sample training dataset."""
    return pd.read_csv(DATA_PATH)


def train_models():
    """Train both models and return them with simple evaluation metrics."""
    data = load_data()
    X = data[FEATURES]
    y_class = data[TARGET_CLASS]
    y_score = data[TARGET_SCORE]

    X_train, X_test, y_class_train, y_class_test = train_test_split(
        X, y_class, test_size=0.2, random_state=42, stratify=y_class
    )
    _, _, y_score_train, y_score_test = train_test_split(
        X, y_score, test_size=0.2, random_state=42
    )

    classifier = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000, random_state=42))
    ])
    regressor = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ])

    classifier.fit(X_train, y_class_train)
    regressor.fit(X_train, y_score_train)

    class_predictions = classifier.predict(X_test)
    score_predictions = regressor.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_class_test, class_predictions),
        "mae": mean_absolute_error(y_score_test, score_predictions),
    }
    return classifier, regressor, metrics


def predict_vibe(answers, classifier, regressor):
    """Return predicted vibe, class probabilities, and estimated energy score."""
    row = pd.DataFrame([answers], columns=FEATURES)
    predicted_vibe = classifier.predict(row)[0]
    probabilities = classifier.predict_proba(row)[0]
    probability_map = dict(zip(classifier.classes_, probabilities))
    energy = float(regressor.predict(row)[0])
    return predicted_vibe, probability_map, max(0.0, min(100.0, energy))
