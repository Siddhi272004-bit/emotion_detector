import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

st.title("Real-time Emotion Detection with streamlit-webrtc")

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(50, 50))
    emotion = "No face detected"

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.8, minNeighbors=20)

        if len(smiles) > 4:
            emotion = "Someone's Really Happy 😊"
        elif len(smiles) > 0:
            emotion = "Someone's Smiling 😃"
        elif w / h < 0.9:
            emotion = "Someone's Sad 😞"
        else:
            emotion = "Neutral 😐"

    cv2.putText(img, emotion, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    return img

webrtc_streamer(key="emotion-detector", video_frame_callback=video_frame_callback)
