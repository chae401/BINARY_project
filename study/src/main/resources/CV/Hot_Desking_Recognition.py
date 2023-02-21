import cv2
import numpy as np
import os
from glob import glob

dir(cv2.face)
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('study\src\main\\resources\CV\\train\\trainer.yml')
faceCascade = cv2.CascadeClassifier('study\src\main\\resources\CV\haarcascade_frontalface_default.xml')



cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1980)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)



# get == 속성 반환
minW = 0.1 * cam.get(cv2.CAP_PROP_FRAME_WIDTH)
minH = 0.1 * cam.get(cv2.CAP_PROP_FRAME_HEIGHT)




# 프레임 받아오고 얼굴 검출
while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=6,
        minSize=(int(minW), int(minH))
    )
    

    # 얼굴 예측
    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 2)

        # 얼굴 예측 & id, confidence == 사용자 id, 확률값 (0에 가까울수록 label과 일치)
        id, confidence = recognizer.predict(gray[y:y+h, x:x+w])


        # 임계값 설정
        # id에 user id 입력
        if confidence < 55 :
            id = id
        else:
            id = "unknown"
        
        confidence = "  {0}%".format(round(100-confidence))
        
        # id와 예측값을 화면에 설정한 폰트로 출력
        cv2.putText(img, str(id), (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.putText(img, str(confidence), (x+5, y+h-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 1)
    

    cv2.imshow('camera', img)
    
    if cv2.waitKey(1) > 0 : break


print(id)
print("\n Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()