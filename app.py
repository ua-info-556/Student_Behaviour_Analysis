# Streamlit Imports for web app
import streamlit as st

# Data Imports for handling dataframes
import pandas as pd

# Visualization Imports for plots
import matplotlib.pyplot as plt
import seaborn as sns

# System Imports for file handling and temp files
import tempfile
import sys
import os
import joblib

# Sklearn Preprocessing and Modeling and Metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Add src to path for file imports
sys.path.append("./src")
from processor import BehaviorProcessor

# OpenCV for video processing
import cv2

# ==========================
# IMPORT TRAINING FUNCTION
# ==========================
from train_models import train_best_engagement_model  # new function in train_models.py


# ==========================
# GLOBAL SETTINGS 
# ==========================
sns.set_theme(style="whitegrid")
st.set_page_config(page_title="Student Behavior Analysis", layout="wide")
plt.rcParams["figure.figsize"] = (6, 4)
plt.rcParams["figure.dpi"] = 120

st.title("🎓 Student Behavior Analysis Dashboard")

# ==========================
# VIDEO SOURCE
# ==========================
source = st.radio("Select Input Source", ["Upload Video"])
video_path = None

if source == "Upload Video":
    uploaded_file = st.file_uploader("Upload classroom/group discussion video", type=["mp4", "mov"])
    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        video_path = tfile.name
        st.success(f"Uploaded video saved to: {video_path}")

# ==========================
# TRAIN MODEL
# ==========================
def train_engagement_model(csv_path="output/behavior_results_aggregated.csv"):
    train_best_engagement_model(dataset_path=csv_path)
    model = joblib.load("models/engagement.pkl")
    le_gaze = joblib.load("models/gaze_enc.pkl")
    le_posture = joblib.load("models/posture_enc.pkl")
    le_emotion = joblib.load("models/emotion_enc.pkl")
    return model, le_gaze, le_posture, le_emotion

# ==========================
# LOAD MODEL
# ==========================
def load_model():
    try:
        return (
            joblib.load("models/engagement.pkl"),
            joblib.load("models/gaze_enc.pkl"),
            joblib.load("models/posture_enc.pkl"),
            joblib.load("models/emotion_enc.pkl"),
        )
    except:
        return None, None, None, None

# ==========================
# BUILD COMBINED DF FROM AGG CSV
# ==========================
def build_combined_df(agg_csv="output/behavior_results_aggregated.csv"):
    if not os.path.exists(agg_csv):
        st.warning("Aggregated CSV not found.")
        return pd.DataFrame()

    df_agg = pd.read_csv(agg_csv)
    gaze_cols = [c for c in df_agg.columns if c.startswith("Gaze_")]
    posture_cols = [c for c in df_agg.columns if c.startswith("Posture_")]
    emotion_cols = [c for c in df_agg.columns if c.startswith("Emotion_")]

    records = []
    for _, row in df_agg.iterrows():
        t = row['Time']
        for col_gaze, col_posture, col_emotion in zip(gaze_cols, posture_cols, emotion_cols):
            student_id = col_gaze.split("_")[1]
            records.append({
                "elapsed": t,
                "student_id": int(student_id),
                "gaze": row[col_gaze] if pd.notna(row[col_gaze]) else "Looking Center",
                "posture": row[col_posture] if pd.notna(row[col_posture]) else "Upright",
                "emotion": row[col_emotion] if pd.notna(row[col_emotion]) else "Neutral",
            })

    combined_df = pd.DataFrame(records)
    if not combined_df.empty:
        combined_df["engagement"] = (
            (combined_df["posture"] == "Upright") &
            (combined_df["gaze"] == "Looking Center") &
            (combined_df["emotion"].isin(["Neutral", "Happy"]))
        ).astype(int)
        combined_df.to_csv("output/behavior_results_aggregated.csv", index=False)

    return combined_df

# ==========================
# RUN ANALYSIS BUTTON
# ==========================
run_analysis = st.button("Run Analysis")

if run_analysis and video_path is not None:
    processor = BehaviorProcessor(video_path, width=640, height=480, frame_skip=2)

    st.info("Initializing video...")

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    progress_bar = st.progress(0.0)
    status_text = st.empty()

    # Count processed frames
    processed_frames = [0]  # use list to allow mutability inside callback

    def update_progress(current_frame, total_frames):
        # Only count processed frames (after frame skip)
        if current_frame % processor.frame_skip == 0:
            processed_frames[0] += 1
            progress = processed_frames[0] / (total_frames / processor.frame_skip)
            progress_bar.progress(min(progress, 1.0))
            status_text.text(f"Processing frame {current_frame}/{total_frames}")

    processor.update_progress = update_progress

    with st.spinner("Analyzing video... Please wait..."):
        processor.process()
        processor.save_results()

    st.success("✅ Video Processing Complete!")

    # Refresh combined data
    st.session_state.combined_df = build_combined_df()
    st.session_state.pop("model", None)

# ==========================
# DISPLAY RESULTS
# ==========================
if "combined_df" in st.session_state and not st.session_state.combined_df.empty:
    combined_df = st.session_state.combined_df

    if "model" not in st.session_state:
        model, le_gaze, le_posture, le_emotion = load_model()
        if model is None:
            with st.spinner("Training engagement model..."):
                model, le_gaze, le_posture, le_emotion = train_engagement_model()
            st.success("✅ Engagement Model Trained!")
        st.session_state.model = (model, le_gaze, le_posture, le_emotion)

    model, le_gaze, le_posture, le_emotion = st.session_state.model

    # Safe encoding
    for col, le in zip(["gaze", "posture", "emotion"], [le_gaze, le_posture, le_emotion]):
        combined_df[f"{col}_label"] = combined_df[col].apply(
            lambda x: le.transform([x])[0] if x in le.classes_ else 0
        )
    combined_df["predicted_engagement"] = model.predict(
        combined_df[["gaze_label", "posture_label", "emotion_label"]]
    )

    # ==========================
    # METRICS
    # ==========================
    st.header("📈 Summary Metrics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("👥 Students", combined_df["student_id"].nunique())
    c2.metric("🎞 Frames", len(combined_df))
    c3.metric("😊 Emotions", combined_df["emotion"].nunique())
    c4.metric("⚡ Engaged Frames", combined_df["predicted_engagement"].sum())

    # ==========================
    # DISTRIBUTION PLOTS
    # ==========================
    st.header("📊 Charts")
    col1, col2, col3, col4 = st.columns(4)
    for col, feature, palette in zip(
        [col1, col2, col3, col4],
        ["gaze", "posture", "emotion", "predicted_engagement"],
        ["Blues", "Oranges", "Greens", "Purples"]
    ):
        with col:
            fig, ax = plt.subplots()
            sns.countplot(x=combined_df[feature], palette=palette, ax=ax)
            ax.set_title(feature.capitalize())
            st.pyplot(fig, use_container_width=True)

    # ==========================
    # STUDENT FILTER
    # ==========================
    selected = st.selectbox(
        "Select Student",
        sorted(combined_df["student_id"].unique())
    )
    student_df = combined_df[combined_df["student_id"] == selected]

    # ==========================
    # TIMELINES
    # ==========================
    st.header(f"⏱ Timelines for Student {selected}")
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.lineplot(x=student_df["elapsed"], y=student_df["emotion"], marker="o", ax=ax)
        ax.set_title("Emotion Timeline")
        st.pyplot(fig, use_container_width=True)
    with col2:
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        sns.lineplot(x=student_df["elapsed"], y=student_df["predicted_engagement"], marker="o", ax=ax2)
        ax2.set_title("Engagement Timeline")
        st.pyplot(fig2, use_container_width=True)

    # ==========================
    # TABLE & DOWNLOAD
    # ==========================
    st.header("📄 Combined Results")
    st.dataframe(combined_df, use_container_width=True)
    st.download_button(
        "⬇️ Download CSV",
        combined_df.to_csv(index=False).encode(),
        "combined_results.csv"
    )

else:
    st.info("No data to display. Please upload a video and run the analysis.")