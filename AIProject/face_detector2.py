from random import randrange
import cv2

#load some pre defined data from xml to trained_face_data variable
trained_face_data = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

#choose an image to detext face
#webcam = cv2.VideoCapture('demo.mp4')
webcam = cv2.VideoCapture(0)

#iterate forever over frames
while True:

    #Read the current frame
    successfu_frame_read, frame= webcam.read()

    #covert it to grey scale
    grayscaled_img = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    #detect faces
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

    #draw rectangle
    for (x, y, w, h) in face_coordinates:
       cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)


    #Display the image with faces
    cv2.imshow("Clever programmer face detector",frame)

    #make code to wait
    key = cv2.waitKey(1)

    #stop if Q key is pressed
    if key ==81 or key == 113:
        break
    
    #stop the webcam
    webcam.release()

    print("Done!")


