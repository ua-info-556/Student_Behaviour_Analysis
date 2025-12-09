import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.dummy import DummyClassifier
import joblib
import streamlit as st

def train_best_engagement_model(dataset_path="output/behavior_results_aggregated.csv"):
    # Load dataset
    df = pd.read_csv(dataset_path)

    # Encode categorical labels
    le_gaze = LabelEncoder()
    le_posture = LabelEncoder()
    le_emotion = LabelEncoder()

    df["gaze_label"] = le_gaze.fit_transform(df["gaze"])
    df["posture_label"] = le_posture.fit_transform(df["posture"])
    df["emotion_label"] = le_emotion.fit_transform(df["emotion"])

    # Engagement label
    df["engagement"] = (
        (df["posture"] == "Upright") &
        (df["gaze"] == "Looking Center") &
        (df["emotion"].isin(["Neutral", "Happy"]))
    ).astype(int)

    X = df[["gaze_label", "posture_label", "emotion_label"]]
    y = df["engagement"]

    # Handle single class case
    if len(y.unique()) < 2:
        dummy_model = DummyClassifier(strategy="constant", constant=y.iloc[0])
        dummy_model.fit(X, y)
        best_model = dummy_model
        best_model_name = f"Dummy ({y.iloc[0]})"
    else:
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Define candidate models
        models = {
            "Random Forest": RandomForestClassifier(n_estimators=120, random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=120, random_state=42),
            "Logistic Regression": LogisticRegression(max_iter=500),
            "SVM": SVC(kernel="rbf", probability=True)
        }

        # Train and evaluate
        results = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            results[name] = (acc, model)
            st.text(f"{name} Accuracy: {acc:.4f}")
            st.text(classification_report(y_test, y_pred))

        # Select best model
        best_model_name = max(results, key=lambda x: results[x][0])
        best_model = results[best_model_name][1]
        st.success(f"✅ Best Engagement Model: {best_model_name} with Accuracy: {results[best_model_name][0]:.4f}")

    # Save model and encoders
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, "models/engagement.pkl")
    joblib.dump(le_gaze, "models/gaze_enc.pkl")
    joblib.dump(le_posture, "models/posture_enc.pkl")
    joblib.dump(le_emotion, "models/emotion_enc.pkl")

    return best_model, le_gaze, le_posture, le_emotion
