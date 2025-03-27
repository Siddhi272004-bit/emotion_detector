import cv2
import numpy as np
import pygame as pg

#load open cv's pre-trained model and smile detectors
face_cascade=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
smile_cascade=cv2.CascadeClassifier(cv2.data.haarcascades+'./haarcascade_smile.xml')

# open webcam
cap=cv2.VideoCapture(0)

# initialized pygame
pg.mixer.init()



while True:
    ret,frame=cap.read()
    if not ret:
        break
    # preprocessing
    # convert frame to grayscale
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    # apply gaussian blur to reduce noise and smooth the image
    blurred=cv2.GaussianBlur(gray,(35,35),0)
    # Apply Thresholding to create a binary image (hand will be white, background black)
    _, thresholded = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)

    # detect faces in the frame
    faces=face_cascade.detectMultiScale(gray,scaleFactor=1.3,minNeighbors=5,minSize=(50,50))

    # DEFAULT EMOTION
    emotion="No face detected"

    for(x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)


        # get the region of interest for smile detection
        roi_gray=gray[y:y+h,x:x+w]

        smiles=smile_cascade.detectMultiScale(roi_gray,scaleFactor=1.8,minNeighbors=20)

        for(sx,sy,sw,sh) in smiles:
            smile_ratio=sw/w
            if smile_ratio>0.2 and sh>20:
                emotion="Someone's SURPRISED"
            elif len(smiles)>4:
                emotion="Someone's Really Happy"
                pg.mixer.music.load("Ilahi_Chorus.mp3")
                pg.mixer.music.play()
                pg.mixer.fadeout(1000)
            elif len(smiles)>0:
                emotion="Someone's Smiling \U0001F606"
                # pg.mixer.music.load("128-Ilahi - Yeh Jawaani Hai Deewani 128 Kbps.mp3")
                # pg.mixer.music.play()
                # pg.mixer.music.fadeout(1000)
            else:
                emotion="Neutral -_-"
        if len(smiles)==0 and w/h <0.9:
            emotion="Someone's Sad"
            pg.mixer.music.load("DieWtihASmile_Chorus.mp3")
            pg.mixer.music.play()
            pg.mixer.fadeout(1000)

        cv2.putText(frame,emotion,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.9,(255,255,255),2)
        # Show the video feed
    cv2.imshow("Real-time Emotion Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
