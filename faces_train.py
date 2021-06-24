import os
import cv2 as cv
import numpy as np

haar_cascade = cv.CascadeClassifier('haar_face.xml')

# p = []
# for i in os.listdir(r'D:\Computer Vision\Face Detection and Recognition\Photos\train'):
#     p.append(i)
# print(p)

people = ['Monica', 'Rachel', 'Pheobe', 'Joey', 'Chandler', 'Ross']
DIR = r'D:\Computer Vision\Face Detection and Recognition\Photos\train'

features = []
labels = []

def create_train():
    for person in people:
        path = os.path.join(DIR, person) 
        label = people.index(person)

        for img in os.listdir(path):
            img_path = os.path.join(path, img)

            img_array = cv.imread(img_path)
            if img_array is None:
                continue 

            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)

            faces_rect = haar_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=4)

            for (x, y, w, h) in faces_rect:
                faces_roi = gray[y:y+h, x:x+w]
                features.append(faces_roi)
                labels.append(label)


create_train()

# print(f'Length of features = {len(features)}')
# print(f'Length of labels = {len(labels)}')

print('Training done ---------------')

features = np.array(features, dtype='object')
labels = np.array(labels)

face_recognizer = cv.face.LBPHFaceRecognizer_create()

# Train the Recognizer on the features list and the labels list
face_recognizer.train(features, labels)

face_recognizer.save('face_trained.yml') # Save the model
np.save('features.npy', features)
np.save('labels.npy', labels)
