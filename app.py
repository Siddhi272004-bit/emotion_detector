import cv2
import numpy as np
import streamlit as st

# Load OpenCV's pre-trained models
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

# Title
st.title("Real-time Emotion Detection")

# Open webcam (inside function)
def detect_emotion():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Error: Could not access the webcam")
        return
    
    frame_placeholder = st.empty()

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture image")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(50, 50))

        emotion = "No face detected"

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.8, minNeighbors=20)

            if len(smiles) > 4:
                emotion = "Someone's Really Happy 😊"
                st.audio("Ilahi_Chorus.mp3", format="audio/mp3")
            elif len(smiles) > 0:
                emotion = "Someone's Smiling 😃"
            elif w/h < 0.9:
                emotion = "Someone's Sad 😞"
                st.audio("DieWtihASmile_Chorus.mp3", format="audio/mp3")
            else:
                emotion = "Neutral 😐"

        frame_placeholder.image(frame, channels="BGR", use_column_width=True)

        if st.button("Stop Camera"):
            break

    cap.release()

# Streamlit Button to Start Webcam
if st.button("Start Emotion Detection"):
    detect_emotion()
