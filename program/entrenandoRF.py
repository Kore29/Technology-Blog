import cv2
import os
import numpy as np

dataPath= './program'
peopleList = os.listdir(dataPath)
print('Person List: ', peopleList)

labels = []
facesData = []
label = 0

for nameDir in peopleList:
    personPath = dataPath + '/' + nameDir
    print('Reading Images')

    for fileName in os.listdir(personPath):
        print('Faces: ', nameDir + '/' + fileName)
        labels.append(label)
        facesData.append(cv2.imread(personPath+'/'+fileName,0))
        image = cv2.imread(personPath+'/'+fileName,0)
        cv2.imshow('image', image)
        cv2.waitKey(10)

    label = label + 1
#print('labels= ', labels)
#print('Tags number 0: ', np.count_nonzero(np.array(labels)==0))
#print('Tags number 1: ', np.count_nonzero(np.array(labels)==1))

#face_recognizer = cv2.face.EigenFaceRecognizer_create()
face_recognizer = cv2.face.FisherFaceRecognizer_create()
print('Training...')
face_recognizer.train(facesData,np.array(labels))

#face_recognizer.write('ModeloEigenFace.xml')
face_recognizer.write('ModeloFisherFace.xml')
print('Storing model....')