import streamlit as st
import time
from PIL import Image
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import cv2
import math
from streamlit_option_menu import option_menu

# Judul Tab
favImage = Image.open("favicon.png")
st.set_page_config(
    page_icon=favImage,
    page_title="Skripsi | Penerjemah Bahasa Isyarat SIBI",
    initial_sidebar_state="auto"  # Collapsed, Extended, Auto
)

# Judul pada Bagian Homepage
st.header = st.markdown("<h1 style='text-align: center;'>Sistem Isyarat Bahasa Indonesia<h1>", unsafe_allow_html=True)

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

offset = 20
imgSize = 1000

folder = "Data/Love"
nameImg = "Love"
counter = 0

FRAME_WINDOW = st.image([]) #frame window
labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L",
          "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W",
          "X", "Y", "Z", "Hai", "I Love You", "Love", "Makasih"]
          
# Menu
menu = ["Home",
        "Deteksi SIBI",
        "About"]

# Side bar
with st.sidebar:
    selected = option_menu(
        menu_title = "Menu",
        options = menu,
        icons = ["house", "code-slash", "info-circle"],
        menu_icon = "cast"
    )

col1, col2, col3 = st.columns(3) #columns

# DETEKSI
if selected == f'{menu[1]}': 
    with col1: #column 1
        st.subheader(f"{menu[1]}")
    st.markdown("Sistem Bahasa Isyarat Indonesia (SIBI)")
    run = st.checkbox("Buka Kamera") #checkbox
    if run :
        while True:
            ret, img = cap.read()
            hands, img = detector.findHands(img)
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
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
                
                cv2.rectangle(img, (x-25,y-offset-70), (x+w+offset, y-offset), (255,255,255), cv2.FILLED)
                cv2.putText(img, labels[index], (x,y-26), cv2.FONT_HERSHEY_COMPLEX, 1.3, (0,0,0), 2)
                cv2.rectangle(img, (x-offset,y-offset), (x+w+offset,y+h+offset), (255,255,255), 4)
            
            FRAME_WINDOW.image(img)
            key = cv2.waitKey(1)
    else:
        pass
# HOME
elif selected == f'{menu[0]}':
    st.subheader(""" Selamat datang di website deteksi kata SIBI """)
    st.text("Anda dapat memilih menu 'Deteksi SIBI' untuk memulai pendeteksian")
    with col1:
        st.write(' ')
        # Membuka Gambar Dengan Nama upload.jpg
    with col2:
        # Menampilkan Gambar kedalam Homepage
        with col1:
            st.write("")
        with col2:
            st.image(Image.open("SIBI2.png"))
        with col3:
            st.write("")
    with col3:
            st.write(' ')
# ABOUT
elif selected == f'{menu[2]}':
    with col1:
        st.write('')
    with col2:
        imgAbout = Image.open("Foto Sidang.jpeg")
        st.image(imgAbout)
    with col3:
        st.write('')
    st.subheader("Hallo")
    st.text(""" Nama saya Rofi Saefurrohman
NIM 15180376
Ini adalah project Skripsi saya
Terima kasih atas ilmunya pak/bu :) """)

with st.sidebar:
    with st.spinner("Bapak Ibu luluskan sayaa..."):
        time.sleep(5)

st.sidebar.text('Rofi Saefurrohman 15180376')

# hide hamburger myapp menu streamlit
hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)