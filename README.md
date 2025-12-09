Student Behavior Analysis
--------------------------------------------------------------------------------------------
This project analyzes classroom or discussion videos to detect student behavior and estimate engagement levels. It uses YOLO models for face and pose detection, tracks students across frames, extracts behavior features (gaze, posture, emotion), and visualizes results in a Streamlit dashboard.


Features
--------------------------------------------------------------------------------------------
  * Face and pose detection using YOLO
  * Student tracking with unique IDs
  * Gaze, posture, and emotion classification
  * Engagement prediction using Random Forest
  * Streamlit dashboard for visualization
  * CSV export of behavior and engagement data


Installation
--------------------------------------------------------------------------------------------
  Clone the repository: 
  
  ```git clone <your-repo-url>```
  ```cd student-behavior-analysis```

  Install dependencies: ```pip install -r requirements.txt``` 

  Place YOLO models in the models folder:

    models/yolov8n-face.pt
    models/yolov8n-pose.pt


How to Run
--------------------------------------------------------------------------------------------
  Streamlit App: Go to the project path on command or terminal then run the below command
  
  ```streamlit run app.py```

  Once Project ran successfully then do the following steps:

    1. Upload a video
    2. Click Run Analysis
    3. View charts, timelines, and engagement metrics


How the system will work:
--------------------------------------------------------------------------------------------
  The system works as follows:

  * Read video
  * Detect faces + poses
  * Track students
  * Extract gaze, posture, emotion
  * Aggregate frame level data
  * Predict engagement
  * Display results in dashboard