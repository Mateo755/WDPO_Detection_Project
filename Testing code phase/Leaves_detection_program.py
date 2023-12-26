import cv2
import numpy as np
import os
import json
from cvzone.ClassificationModule import Classifier
import tensorflow as tf



def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver


def getCountour(img):
    countours, hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    aspen = 0
    birch = 0
    hazel = 0
    maple = 0
    oak = 0
    for cnt in countours:
        area = cv2.contourArea(cnt)
        #print(area)
        if area > 80:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(imgCountour, (x, y), (x + w, y + h), (0, 255, 0), 2)
            roi = imgCountour[y:y + h, x:x + w]


            prediction, index = classifier.getPrediction(roi,draw=False)
            leaf = ResultMap[index]

            #print('####' * 10)
            #print(index)
            #print('Prediction is: ', ResultMap[index])

            cv2.putText(imgCountour, ResultMap[index], (x,y), cv2.FONT_HERSHEY_COMPLEX, 2 ,(255,0,255),1)

            match leaf:
                case "aspen":
                    aspen += 1
                case "birch":
                    birch += 1
                case "hazel":
                    hazel += 1
                case "maple":
                    maple += 1
                case "oak":
                    oak += 1
                case _:
                    print("Undefined leaf.")

    result = {'aspen': aspen, 'birch': birch, 'hazel': hazel, 'maple': maple, 'oak': oak}
    print(result)

    with open(result_json_path, 'w') as json_file:
        json.dump(result, json_file, indent=2)



TrainClasses =  {'aspen': 0, 'birch': 1, 'hazel': 2, 'maple': 3, 'oak': 4}

# Storing the face and the numeric tag for future reference
ResultMap={}
for idx, leaf in zip(TrainClasses.values(),TrainClasses.keys()):
    ResultMap[idx]=leaf


input_dir = '../data'
result_json_path = "result.json"

# Uzyskaj listę plików w folderze
files = os.listdir(input_dir)

# Zlicz liczbę plików
num_files = len(files)

input_path = os.path.join(input_dir, '0015.jpg')

classifier = Classifier('../Model/leaves_classifier_model.h5','../Model/labels.txt')

img = cv2.imread(input_path)
#img = cv2.resize(img, dsize=(1000, 800), interpolation=cv2.INTER_LANCZOS4)


imgCountour = img.copy()

imgBlank = np.zeros_like(img)

imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(imgGray,250,255,cv2.THRESH_BINARY_INV)

imgBlur = cv2.GaussianBlur(thresh,(5,5),0)
imgCanny = cv2.Canny(imgBlur,120,120)

getCountour(imgCanny)

imgStack = stackImages(0.5, ([imgCountour]))
cv2.imshow("ImageStack", imgStack)
cv2.waitKey(0)

