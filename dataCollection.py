import time
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math

cap = cv2.VideoCapture(2)
detector = HandDetector(maxHands=1)

offset = 20
imgSize = 400

folder = "Data/N"
nameImg = "N"
counter = 0

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
            k = imgSize/h # constant
            wCal = math.ceil(k*w) # width calculated, math.ceil(if 3.2 or 3.5 it will be 4)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize-wCal)/2) # width Gap to find gap that make image shape to the center
            # imgWhite[0:imgResizeShape[0], 0:imgResizeShape[1]] = imgResize
            imgWhite[:, wGap:wCal+wGap] = imgResize

        else :
            k = imgSize/w # constant
            hCal = math.ceil(k*h) # width calculated, math.ceil(if 3.2 or 3.5 it will be 4)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize-hCal)/2) # width Gap to find gap that make image shape to the center
            imgWhite[hGap:hCal+hGap, :] = imgResize

        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("imgWhite", imgWhite)

    cv2.imshow("skripsi", img)
    key = cv2.waitKey(1)

    # === collecting images ===
    if key == ord("s"):
        counter += 1
        cv2.imwrite(f'{folder}/{nameImg}_{counter}{time.time()}.jpg', imgWhite)
        print(counter)