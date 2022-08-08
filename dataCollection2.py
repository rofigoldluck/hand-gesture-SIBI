import time
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math

cap = cv2.VideoCapture(2)
detector = HandDetector(maxHands=2)

offset = 20
imgSize = 300

folder = "Data/Test"
nameImg = "Test"
counter = 0

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    # === crop detector ===
    if hands :
        hand1 = hands[0]
        x1, y1, w1, h1 = hand1['bbox']

        imgWhite1 = np.ones((imgSize, imgSize, 3), np.uint8)*255
        imgCrop1 = img[y1-offset:y1 + h1+offset, x1-offset:x1 + w1+offset]

        imgCropShape = imgCrop1.shape

        # if height > 300 width scratch, also if width > 300 height scracth
        aspectRatio = h1/w1

        if len(hands) == 2:
            hand2 = hands[1]
            x2, y2, w2, h2 = hand2['bbox']

            imgWhite2 = np.ones((imgSize*2, imgSize*2, 3), np.uint8)*255
            if x2 < 100 :
                imgCrop2 = img[y1-offset:y2 + h1+offset, x1-offset:x2 + w1+offset]
            else :
                imgCrop2 = img[y2-offset:y1 + h2+offset, x2-offset:x1 + w2+offset]
            
            imgCropShape2 = imgCrop2.shape

        #     # if height > 300 width scratch, also if width > 300 height scracth
        #     aspectRatio2 = h2/w2
        #     if aspectRatio2 > 1 :
        #         k2 = imgSize/h2 # constant
        #         wCal2 = math.ceil(k2*w2) # width calculated, math.ceil(if 3.2 or 3.5 it will be 4)
        #         imgResize2 = cv2.resize(imgCrop2, (wCal2, imgSize))
        #         imgResizeShape2 = imgResize2.shape
        #         wGap2 = math.ceil((imgSize-wCal2)/2) # width Gap to find gap that make image shape to the center
        #         # imgWhite[0:imgResizeShape[0], 0:imgResizeShape[1]] = imgResize
        #         imgWhite2[:, wGap2:wCal2+wGap2] = imgResize2

        #     else :
        #         k2 = imgSize/w2 # constant
        #         hCal2 = math.ceil(k2*h2) # width calculated, math.ceil(if 3.2 or 3.5 it will be 4)
        #         imgResize2 = cv2.resize(imgCrop2, (imgSize, hCal2))
        #         imgResizeShape2 = imgResize2.shape
        #         hGap2 = math.ceil((imgSize-hCal2)/2) # width Gap to find gap that make image shape to the center
        #         imgWhite2[hGap2:hCal2+hGap2, :] = imgResize2

        # if aspectRatio > 1 :
        #     k1 = imgSize/h1 # constant
        #     wCal1 = math.ceil(k1*w1) # width calculated, math.ceil(if 3.2 or 3.5 it will be 4)
        #     imgResize1 = cv2.resize(imgCrop1, (wCal1, imgSize))
        #     imgResizeShape1= imgResize1.shape
        #     wGap1 = math.ceil((imgSize-wCal1)/2) # width Gap to find gap that make image shape to the center
        #     # imgWhite[0:imgResizeShape[0], 0:imgResizeShape[1]] = imgResize
        #     imgWhite1[:, wGap1:wCal1+wGap1] = imgResize1

        # else :
        #     k1 = imgSize/w1 # constant
        #     hCal1 = math.ceil(k1*h1) # width calculated, math.ceil(if 3.2 or 3.5 it will be 4)
        #     imgResize1 = cv2.resize(imgCrop1, (imgSize, hCal1))
        #     imgResizeShape1 = imgResize1.shape
        #     hGap1 = math.ceil((imgSize-hCal1)/2) # width Gap to find gap that make image shape to the center
        #     imgWhite1[hGap1:hCal1+hGap1, :] = imgResize1

            cv2.imshow("ImageCrop", imgCrop2)
            cv2.imshow("imgWhite", imgWhite2)
            # print(imgCropShape2)
            print(f"{x2}, {y2}, {w2}, {h2}")

    cv2.imshow("skripsi", img)
    key = cv2.waitKey(1)

    # === collecting images ===
    if key == ord("s"):
        counter += 1
        cv2.imwrite(f'{folder}/{nameImg}_{counter}{time.time()}.jpg', imgWhite2)
        print(counter)
    # i = 0
    # for i in range(int(300)) :
    #     i += 1
    #     cv2.imwrite(f'{folder}/{nameImg}_{counter}{time.time()}.jpg', img)
    #     print(i)
    # break