import cv2
import os
import numpy as np
from PIL import Image
from glob import glob


faceCascade = cv2.CascadeClassifier('study\src\main\\resources\CV\haarcascade_frontalface_default.xml')


cam = cv2.VideoCapture(0)

cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)



face_id = input('\n enter user id end press <return> ==> ')
print('\n Initializing face capture. Look the camera and waith . . .')


new_face_id_dir = os.path.join('study\src\main\\resources\CV\Face_Dataset\\' + str(face_id))

print(new_face_id_dir)

if not os.path.exists(new_face_id_dir):
    os.makedirs(new_face_id_dir)



cnt = 0



if cam.isOpened() == False:
    print("Failed to open camera !!!")
    exit()



while cam.isOpened():


    ret, frame = cam.read()



    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)



    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor = 1.2,
        minNeighbors = 6,
        minSize = (20, 20)
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        cnt += 1
        percent = str(cnt) + '%'
        
        cv2.imwrite(new_face_id_dir + '\\' + str(cnt) + '.jpg', gray[y:y+h, x:x+w])
        cv2.putText(frame, str(percent), (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)


    cv2.imshow('Save FaceId Data', frame)


    if cv2.waitKey(1) > 0 : break
    elif cnt >= 100 : break

print("\n [INFO] Exiting Program and cleanup stuff")

cam.release()
cv2.destroyAllWindows()






path = 'study\src\main\\resources\CV\Face_Dataset\\'




dir(cv2.face)
recognizer = cv2.face.LBPHFaceRecognizer_create()




def getImagesAndLabels(path):

    image_paths = glob(os.path.join(path, '*', '*'))




    for i in range(len(image_paths)):
        print(image_paths[i])


    faceSamples = []
    ids = []

    for image_path in image_paths:

        print('image_path = ', image_path)

        img = Image.open(image_path).convert('L')
        img_numpy = np.array(img, 'uint8')

        print("point1 Pass\n")

        id = int(os.path.split(image_path)[-2].split('\\')[-1])

        print(id)

        faces = faceCascade.detectMultiScale(img_numpy)

        for(x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)

    return faceSamples, ids


print('\n Training faces. It will take a few seconds.` Wait ...')

faces, ids = getImagesAndLabels(path)

recognizer.train(faces, np.array(ids))

recognizer.write('study\src\main\\resources\CV\\train\\trainer.yml')

print(f"\n {len(np.unique(ids))} faces trained. Exiting Program")