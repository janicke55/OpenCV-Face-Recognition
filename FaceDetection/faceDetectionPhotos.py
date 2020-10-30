'''
Haar Cascade Face detection with OpenCV  
    Based on tutorial by pythonprogramming.net
    Visit original post: https://pythonprogramming.net/haar-cascade-face-eye-detection-python-opencv-tutorial/  
Adapted by Marcelo Rovai - MJRoBot.org @ 7Feb2018 
'''
import time
import numpy as np
import cv2, os

# multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades
# faceCascade = cv2.CascadeClassifier('Cascades/haarcascade_frontalface_default.xml')
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")



for dirname, _, filenames in os.walk("C:\\Users\\Jany\\Documents\\GitHub\\OpenCV-Face-Recognition\\FacialRecognition\datasetTemp\TestingPhotos"):
    for filename in filenames:

        img = cv2.imread(os.path.join(dirname, filename))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(20, 20)
        )

        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]

        cv2.imshow('Found Faces', cv2.resize(img, (1920, 1080)))
        k = cv2.waitKey(10) & 0xff  # Press 'ESC' for exiting video

        while k != 32:  # space
            if k == 27:  # esc
                break
            k = cv2.waitKey(10) & 0xff  # Press 'ESC' for exiting video
            time.sleep(0.1)


