import cv2 as cv
import numpy as np
import dlib

cap = cv.VideoCapture(0)

detector = dlib.get_frontal_face_detector()

predictor = dlib.shape_predictor("C:\\Users\\user\\Desktop\\project\\shape_predictor_68_face_landmarks.dat")

while True:
    ret, frame = cap.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)  # 因為gray只有一個channel所以比較好處理
    
    faces =  detector(gray)
    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        #cv.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)  #這個rectangel是放在哪一張圖片上，第一個點，第二個點，顏色，thickniss

        landmarks = predictor(gray, face)
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv.circle(frame, (x, y), 4, (255, 0, 0), -1)

        
        
    #print(faces)  # face 在影像中的位置，會有兩個point，top left of the screen where the faces is detected
                  # another is right bottom

    
    
    
    frame = cv.flip(frame, 1)
    
    cv.imshow('me', frame)
    
    key = cv.waitKey(1)
    if key == 27:
        break