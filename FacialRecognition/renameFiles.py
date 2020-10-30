
import os
import cv2

face_id = input('\n enter ID to replace ==>  ')
count = 35

path = os.getcwd() + "\\datasetLearn\\faces3"

for dirname, _, filenames in os.walk(path):
    for filename in filenames:
        img = cv2.imread(os.path.join(dirname, filename))   # Fotky namiesto videa ...
        cv2.imwrite(path + '\\user.' + str(face_id) + '.' + str(count) + ".jpg", img)
        count += 1
