import cv2
from tensorflow import keras
from matplotlib import pyplot as plt
import numpy as np
import time 
import os

curr_path = os.path.dirname(__file__)

def start():

    model = keras.models.load_model(curr_path+"/roadseg_epoch30_dataset867.h5",compile=False)
    cap = cv2.VideoCapture(curr_path+"/testing_video.mp4")

    while cap.read():
        start = time.time()
        ret, frame = cap.read()
        print(frame.shape)
        test_img = cv2.resize(frame,(512,512))
        rot_img =  cv2.rotate(test_img,cv2.ROTATE_180)
        cv2.imshow("frame",test_img)
        frame = np.expand_dims(test_img, axis = 0)
        prediction = model.predict(frame)
        prediction_image = prediction.reshape((512,512))
        cv2.imshow("prediction",prediction_image)
        end = time.time()
        d = end - start
        fps = 1/d
        print(fps)
        if cv2.waitKey(1) == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


if "__main__" == __name__ : 
    start()
    