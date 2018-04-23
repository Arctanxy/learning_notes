import cv2 
import numpy as np 

face_cascade = cv2.CascadeClassifier(r'G:\opencv\sources\data\haarcascades\haarcascade_frontalface_default.xml')

eye_cascade = cv2.CascadeClassifier(r'G:\opencv\sources\data\haarcascades\haarcascade_eye.xml')

#cap = cv2.VideoCapture(1)

#while True:
img = cv2.imread("H:/learning_notes/study/opencv/Diablo.jpg")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray,1.3,5)
for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)#找到脸之后画个长方形框出来
    roi_gray = gray[y:y+h,x:x+w]
    roi_color = img[y:y+h,x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)#在脸的范围找眼镜
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew),(ey+eh),(0,255,0),2)

cv2.imshow('img',img)
cv2.waitKey(0)

#cap.release()
#cv2.destroyAllWindows()


