import cv2
import dlib
from imutils import face_utils
from keras.models import load_model
import numpy as np
import pandas as pd
loaded_model=load_model("multiclass1-2.h5")
maxval = pd.read_csv("datamax.csv")
import time
import csv

# define a video capture object
vid = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
f=open("time_data.csv","w")
writer=csv.writer(f)
c = 0
while (True):

    # Capture the video frame
    # by frame
    ret, frame = vid.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Display the resulting frame
    rects = detector(gray, 0)
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        start_time=time.time()
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        # print(shape)
        x = []
        for i in shape:
            x.append(i[0])
            x.append(i[1])
        for i in range(0,136):
            x[i] = float(x[i])/float(maxval.loc[i])
        predicted = loaded_model.predict(np.array([x,]))
        names =  ["blinking","nonsleepy","sleepy","yawning","eyes open","eyes closed","head still","head nodding","head looking aside","mouth still","mouth yawn","mouth talking"]
        # convert dlib's rectangle to a OpenCV-style bounding box
        # [i.e., (x, y, w, h)], then draw the face bounding box
        # print(shape)
        labels=[]
        for i,p in enumerate(predicted):
        	if predicted[i][0]>0.5:
        		labels.append(names[i])


        (x, y, w, h) = face_utils.rect_to_bb(rect)
        print(x, y, w, h)
        cv2.rectangle(gray, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # show the face number
        cv2.putText(gray, "Face #{}".format(i + 1), (x - 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        for i,l in enumerate(labels):
        	cv2.putText(gray,l,(100,i*50+100),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)

        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image
        for (x, y) in shape:
            cv2.circle(gray, (x, y), 1, (0, 0, 255), -1)

        if "mouth talking" in labels:
            cv2.imwrite("MouthTalk/"+str(c)+".png",gray)
            c+=1
    cv2.imshow('frame', gray)

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
vid.release()
f.close()
# Destroy all the windows
cv2.destroyAllWindows()