import cv2
import numpy as np
import face_recognition

imgMain = face_recognition.load_image_file('ImagesBasic/eion.jpg')
imgMain = cv2.cvtColor(imgMain,cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file('ImagesBasic/bill-gates.jpg')
imgTest = cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(imgMain)[0]
encodeMe = face_recognition.face_encodings(imgMain)[0]
cv2.rectangle(imgMain,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)

faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2)

results = face_recognition.compare_faces([encodeMe],encodeTest)
faceDis = face_recognition.face_distance([encodeMe],encodeTest)
print(results,faceDis)
cv2.putText(imgTest,f'{results} {round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

cv2.imshow('main',imgMain)
cv2.imshow('test',imgTest)
cv2.waitKey(0)
