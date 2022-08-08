import time
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

cap = cv2.VideoCapture(2)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

offset = 20
imgSize = 400

folder = "Data/Love"
nameImg = "Love"
counter = 0

labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L",
          "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W",
          "X", "Y", "Z", "Hai", "I Love You", "Love", "Makasih"]

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    # === crop detector ===
    if hands :
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255
        imgCrop = img[y-offset:y + h+offset, x-offset:x + w+offset]

        imgCropShape = imgCrop.shape

        # if height > 300 width scratch, also if width > 300 height scracth
        aspectRatio = h/w

        if aspectRatio > 1 :
            try :
                k = imgSize/h # constant
                wCal = math.ceil(k*w) # width calculated, math.ceil(if 3.2 or 3.5 it will be 4)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                imgResizeShape = imgResize.shape
                wGap = math.ceil((imgSize-wCal)/2) # width Gap to find gap that make image shape to the center
                # imgWhite[0:imgResizeShape[0], 0:imgResizeShape[1]] = imgResize
                imgWhite[:, wGap:wCal+wGap] = imgResize
                prediction, index = classifier.getPrediction(imgWhite)
            except :
                prediction, index = classifier.getPrediction(img)

        else :
            try :
                k = imgSize/w # constant
                hCal = math.ceil(k*h) # width calculated, math.ceil(if 3.2 or 3.5 it will be 4)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                imgResizeShape = imgResize.shape
                hGap = math.ceil((imgSize-hCal)/2) # width Gap to find gap that make image shape to the center
                imgWhite[hGap:hCal+hGap, :] = imgResize
                prediction, index = classifier.getPrediction(imgWhite)
            except :
                prediction, index = classifier.getPrediction(img)
        
        cv2.rectangle(img, (x-25,y-offset-70), (x+w+offset, y-offset), (255,0,255), cv2.FILLED)
        cv2.putText(img, labels[index], (x,y-26), cv2.FONT_HERSHEY_COMPLEX, 1.3, (255,255,255), 2)
        cv2.rectangle(img, (x-offset,y-offset), (x+w+offset,y+h+offset), (255,0,255), 4)

        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("imgWhite", imgWhite)

    cv2.imshow("skripsi", img)
    key = cv2.waitKey(1)