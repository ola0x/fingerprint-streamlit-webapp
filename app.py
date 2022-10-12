import os
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from FingerprintImageEnhancer import FingerprintImageEnhancer

from crop_finger import get_cropped_finger
from inference import initialization

def main():
    # Initialization
    cfg, predictor = initialization()
    image_enhancer = FingerprintImageEnhancer() 

    # Streamlit initialization
    html_temp = """
        <div style="background-color:green;padding:5px">
        <h2 style="color:white;text-align:center;">Finger Print </h2>
        </div>

        """
    st.markdown(html_temp,unsafe_allow_html=True)

    # Retrieve image
    uploaded_img = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
    if uploaded_img is not None:
        file_bytes = np.asarray(bytearray(uploaded_img.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        # Detection code
        cropped_finger, full_img = get_cropped_finger(img,
                                            predictor,return_mapping=True,
                                            resize = (300,int(300*.7)))

        gray_finger = cv2.cvtColor(np.ascontiguousarray(cropped_finger), cv2.COLOR_BGR2GRAY)    

        # Apply 3x3 and 7x7 Gaussian blur
        low_sigma = cv2.GaussianBlur(gray_finger,(3,3),0)
        high_sigma = cv2.GaussianBlur(gray_finger,(5,5),0)

        # Calculate the DoG by subtracting
        dog = low_sigma - high_sigma 

        out = image_enhancer.enhance(dog)
        # out = np.rot90(np.array(out), k=1)
        # image_enhancer.save_enhanced_image("save.jpg")
        out = Image.fromarray(out)                        

        st.image(out, caption='Enhanced FingerPrint Image')       


if __name__ == '__main__':
    main()