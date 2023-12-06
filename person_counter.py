import cv2 # opencv --> image processing libaries
import datetime
import imutils # image processing
import numpy as np
from nms import non_max_suppression_fast
from centroidtracker import CentroidTracker
from datetime import  date, time

#deep learning model, caffe is similar tensorflow 
detector = cv2.dnn.readNetFromCaffe(prototxt="MobileNetSSD_deploy.prototxt", caffeModel="MobileNetSSD_deploy.caffemodel")

import pandas as pd

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

tracker = CentroidTracker(maxDisappeared=80, maxDistance=90)


def main():
    #cap = cv2.VideoCapture(0)
    cap=cv2.VideoCapture('test_video.mp4')
    #cap=cv2.VideoCapture('cat.mp4')
    fps_start_time = datetime.datetime.now()
    fps = 0
    total_frames = 0
    lpc_count = 0
    opc_count = 0
    object_id_list = []
    #data=['id','intime','outtime']
    my_dict = {"Counter":[],"In_time":[]}
    
    while True:
        ret, frame = cap.read() # image read frame by frame
        frame = imutils.resize(frame, width=600) # image resize to width 600
        #total_frames = total_frames + 1
        #print(frame.shape)
        (H, W) = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)#image,scalefactor,size,mean to get 4 dim array
        get_hour=datetime.datetime.now()
        detector.setInput(blob) 
        person_detections = detector.forward()
        rects = []
        for i in np.arange(0, person_detections.shape[2]):
            confidence = person_detections[0, 0, i, 2]
            if confidence > 0.5:
                idx = int(person_detections[0, 0, i, 1])

                if CLASSES[idx] != "person" and CLASSES[idx] != "cat":
                    continue

                person_box = person_detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                (startX, startY, endX, endY) = person_box.astype("int")
                rects.append(person_box)

        boundingboxes = np.array(rects)
        boundingboxes = boundingboxes.astype(int)
        rects = non_max_suppression_fast(boundingboxes, 0.3)

        objects = tracker.update(rects)
        for (objectId, bbox) in objects.items():
            x1, y1, x2, y2 = bbox
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            #text = "ID: {}".format(objectId)
            #cv2.putText(frame, text, (x1, y1-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
            
        #    if(datetime.datetime.now().minute>0):
               #print('hello')
            if objectId not in object_id_list:
                  my_dict["Counter"].append(objectId)
                  now = datetime.datetime.now()
                  #hr = datetime.strftime("11/03/22 14:23", "%d/%m/%y %H:%M") 
                  time = now.strftime("%y-%m-%d %H:%M:%S")
                  #print(hr) 
                  my_dict["In_time"].append(str(time))

                
                  object_id_list.append(objectId)
            
     

        #fps_end_time = datetime.datetime.now()
        #time_diff = fps_end_time - fps_start_time
        #if time_diff.seconds == 0:
        #    fps = 0.0
        #else:
        #    fps = (total_frames / time_diff.seconds)

        #fps_text = "FPS: {:.2f}".format(fps) 

        #cv2.putText(frame, fps_text, (5, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)

        #lpc_count = len(objects)
        opc_count = len(object_id_list)

        #lpc_txt = "LPC: {}".format(lpc_count)
        opc_txt = "Counter: {}".format(opc_count)
        #dict2.update({: "Scala"})

        #cv2.putText(frame, lpc_txt, (5, 60), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
        cv2.putText(frame, opc_txt, (5, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1)

        cv2.imshow("Application", frame)
        #frame1 = frame2
        key = cv2.waitKey(1)
        if key == ord('q'): 
            print(my_dict)
            #print(opc_txt)
            df=pd.DataFrame.from_dict(my_dict)
            #df.set_index('In_time', inplace=True)
            #print(df[df["In_time"]<"16:46:00"])
            df.to_csv('person_counter.csv', index=False)
            #df=df.loc[df['In_time']]
            #df=df[(df.index.hour>13)]
            #print(df)
            #ydf.loc[df['In_time'].dt.time > time(17,00)]
            break

    cv2.destroyAllWindows()


main()
