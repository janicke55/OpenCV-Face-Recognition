''''
Capture multiple Faces from multiple users to be stored on a DataBase (dataset_to_train directory)
	==> Faces will be stored on a directory: dataset_to_train/ (if does not exist, pls create one)
	==> Each face will have a unique numeric integer ID as 1, 2, 3, etc                       

Based on original code by Anirban Kar: https://github.com/thecodacus/Face-Recognition    

Developed by Marcelo Rovai - MJRoBot.org @ 21Feb18    

'''

import cv2
import os

# C:\Users\Jany\Documents\GitHub\OpenCV-Face-Recognition\FacialRecognition\datasetLearn\FacesOnly
save_path = os.getcwd() + "\\datasetLearn\\faces"

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# For each person, enter one numeric face id
face_id = input('\n enter user id end press <return> ==>  ')
count = 0


for dirname, _, filenames in os.walk(os.getcwd() + "\\datasetLearn\\TestingPhotos"):
    for filename in filenames:

        img = cv2.imread(os.path.join(dirname, filename))   # Fotky namiesto videa ...
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_detector.detectMultiScale(gray, 1.3, 5)

        for (x,y,w,h) in faces:

            cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
            count += 1
            # Save the captured image into the datasets folder
            actual_file = save_path + str(face_id) + '\\user.' + str(face_id) + '.' + str(count) + ".jpg"
            cv2.imwrite(actual_file, gray[y:y + h, x:x + w])
            # cv2.imwrite("dataset_to_train/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])

            #cv2.imshow('image', img)

# Do a bit of cleanup
print("\n [INFO] Faces generated Successfully")



