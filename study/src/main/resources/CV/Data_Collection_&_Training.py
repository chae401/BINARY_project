# dataset 폴더 생성 후 실행
import cv2
import os
import numpy as np
from PIL import Image
from glob import glob

# 분석 모델
faceCascade = cv2.CascadeClassifier('study\src\main\\resources\CV\haarcascade_frontalface_default.xml')



# 카메라 세팅
cam = cv2.VideoCapture(0) # 내장카메라 == 0, 외장 == 1
# 3:4 카메라
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)



face_id = input('\n enter user id end press <return> ==> ') # 사용자 id 입력
print('\n Initializing face capture. Look the camera and waith . . .')


# id에 해당하는 이름으로 폴더 생성
new_face_id_dir = os.path.join('study\src\main\\resources\CV\Face_Dataset\\' + str(face_id))

print(new_face_id_dir)

if not os.path.exists(new_face_id_dir):
    os.makedirs(new_face_id_dir)


# 사진 count
cnt = 0


# 영상 처리 및 출력


if cam.isOpened() == False:
    print("Failed to open camera !!!")
    exit()



while cam.isOpened():

    # 카메라 상태 및 프레임
    # 카메라가 정상이면 ret == True

    ret, frame = cam.read()



    # haar cascade는 흑백으로 처리
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)



    # 추후에 설명 추가
    faces = faceCascade.detectMultiScale(
        gray, # 검출 하고 싶은 원본 이미지
        scaleFactor = 1.2, # 검색 윈도우 확대 비율, 1보다 커야 함
        minNeighbors = 6,  # 얼굴 사이 최소 간격 (픽셀값)
        minSize = (20, 20) # 얼굴 최소 크기 / 이 사이즈보다 작으면 무시
    )

    # 얼굴 rectangle
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        cnt += 1
        percent = str(cnt) + '%'
        
        cv2.imwrite(new_face_id_dir + '\\' + str(cnt) + '.jpg', gray[y:y+h, x:x+w])
        cv2.putText(frame, str(percent), (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)


    cv2.imshow('Save FaceId Data', frame)

    # 종료 조건
    # if cv2.waitKey(1) > 0 : break 필수 입력 ! => 아니면 카메라 제대로 작동 X
    if cv2.waitKey(1) > 0 : break
    elif cnt >= 100 : break

print("\n [INFO] Exiting Program and cleanup stuff")

cam.release() #메모리 해제
cv2.destroyAllWindows()#모든 윈도우 창 닫기





path = 'study\src\main\\resources\CV\Face_Dataset\\'

# 그냥 recognizer = cv2.face.LBPHFaceRecognizer_create() 이것만 실행하면
# 계속 cv2에 face 모듈이 존재하지 않는다고 오류 발생
# print(dir(cv2.face)) 코드 추가
# 코드 추가하니까 recognizer = cv2.face.LBPHFaceRecognizer_create() 잘 실행 됨
dir(cv2.face)
recognizer = cv2.face.LBPHFaceRecognizer_create()




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
        faces = faceCascade.detectMultiScale(img_numpy)

        for(x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)

    return faceSamples, ids


print('\n Training faces. It will take a few seconds.` Wait ...')

faces, ids = getImagesAndLabels(path)

recognizer.train(faces, np.array(ids)) #학습

recognizer.write('study\src\main\\resources\CV\\train\\trainer.yml')

print(f"\n {len(np.unique(ids))} faces trained. Exiting Program")