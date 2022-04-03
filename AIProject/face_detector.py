from random import randrange
import cv2

#load some pre defined data from xml to trained_face_data variable
trained_face_data = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

#choose an image to detext face
img = cv2.imread("RDJ2.jpg")

#covert it to grey scale
grayscaled_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#detect faces
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

#draw rectangle
for (x, y, w, h) in face_coordinates:
    cv2.rectangle(img, (x,y), (x+w, y+h), (randrange(256),randrange(256),randrange(256)), 2)

#Display the image with faces
cv2.imshow("Clever programmer face detector",img)

#make code to wait
cv2.waitKey()

print("Done!")
