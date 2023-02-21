import cv2
import numpy as np
from PIL import Image
from glob import glob
import os


path = 'study\src\main\\resources\CV\Face_Dataset\\'

# 그냥 recognizer = cv2.face.LBPHFaceRecognizer_create() 이것만 실행하면
# 계속 cv2에 face 모듈이 존재하지 않는다고 오류 발생
# print(dir(cv2.face)) 코드 추가
# 코드 추가하니까 recognizer = cv2.face.LBPHFaceRecognizer_create() 잘 실행 됨
dir(cv2.face)
recognizer = cv2.face.LBPHFaceRecognizer_create()


detector = cv2.CascadeClassifier('study\src\main\\resources\CV\haarcascade_frontalface_default.xml')


def getImagesAndLabels(path):

    image_paths = glob(os.path.join(path, '*', '*'))
    # listdir == 해당 디렉토리 내 파일 리스트
    # path + file Name == 경로 list 만들기



    for i in range(len(image_paths)):
        print(image_paths[i])


    faceSamples = []
    ids = []

    for image_path in image_paths: # 각 파일마다
        # 흑백 변환
        # 8 bits pixel 이미지로 변환 => 0 ~ 255의 수로 표현 가능한 흑백 이미지 생성
        print('image_path = ', image_path)

        img = Image.open(image_path).convert('L') # L : 8 bit pixel, bw
        img_numpy = np.array(img, 'uint8')

        print("point1 Pass\n")

        id = int(os.path.split(image_path)[-2].split('\\')[-1])

        print(id)

        # 학습을 위한 얼굴 샘플
        faces = detector.detectMultiScale(img_numpy)

        for(x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)

    return faceSamples, ids


print('\n Training faces. It will take a few seconds.` Wait ...')

faces, ids = getImagesAndLabels(path)

recognizer.train(faces, np.array(ids)) #학습

recognizer.write('study\src\main\\resources\CV\\train\\trainer.yml')

print(f"\n {len(np.unique(ids))} faces trained. Exiting Program")