''''
Real Time Face Recogition
	==> Each face stored on dataset_to_train/ dir, should have a unique numeric integer ID as 1, 2, 3, etc
	==> LBPH computed dataModel (trained faces) should be on trainer/ dir
Based on original code by Anirban Kar: https://github.com/thecodacus/Face-Recognition    

Developed by Marcelo Rovai - MJRoBot.org @ 21Feb18  

'''

import cv2
import numpy as np
import os
import time

model_path = os.getcwd() + "\\data_model\\trainer.yml"

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(model_path)
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);

font = cv2.FONT_HERSHEY_SIMPLEX

#iniciate id counter
id = 0

minW = 192
minH = 108

# names related to ids: example ==> Marcelo: id=1,  etc
# names = ['Jany']
names = ['Neznama osoba', 'Maja liptak', 'Sona Pircakova', 'Jany Pircak', 'Z', 'W']


for dirname, _, filenames in os.walk(os.getcwd() + "\\datasetLearn\\TestingPhotos"):
    for filename in filenames:

        img = cv2.imread(os.path.join(dirname, filename))   # Fotky namiesto videa ...

        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor = 1.2,
            minNeighbors = 5,
            minSize = (int(minW), int(minH)),
           )

        for(x,y,w,h) in faces:

            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

            id, error = recognizer.predict(gray[y:y+h,x:x+w])

            # Check if confidence is less them 100 ==> "0" is perfect match
            if (error < 30):
                id = names[id]
                confidence = "  {0}%".format(round(100 - error))
            else:
                id = names[0]
                confidence = "  {0}%".format(round(100 - error))

            cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
            cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)

        # cv2.imshow('camera',img)
        cv2.imshow('camera',cv2.resize(img, (1920, 1080)))
        k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video


        while k != 32:  #space
            if k == 27: #esc
                break
            k = cv2.waitKey(10) & 0xff  # Press 'ESC' for exiting video
            time.sleep(0.01)



# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")

