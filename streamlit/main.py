import math

import streamlit as st
import numpy as np
import tensorflow as tf
import cv2

class_names = ['Benign', 'Early', 'Pre', 'Pro']

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

@st.cache_resource
def load_model():
    return tf.saved_model.load('googlenet_model/clahe_balanced_model').signatures['serving_default']

model = load_model()

st.title('Klasifikasi leukemia limfoblastik akut dengan arsitektur googlenet')
images_uploaded = st.file_uploader("Upload file", type=['jpg', 'png'], accept_multiple_files=True)



for i in range(math.ceil(len(images_uploaded) / 3)):
    cols = min(3, len(images_uploaded) - (i*3))
    row = st.columns(cols)

    for j in range(cols):
        row[j].image(images_uploaded[i*3 + j], width=224)

        # read image
        arr = np.asarray(bytearray(images_uploaded[i*3 + j].read()), dtype=np.uint8)
        print(arr)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        
        # clahe preprocessing
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])

        img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        # resize and normalization (from int to float 0-1)
        img = cv2.resize(img, (224, 224))
        img = tf.convert_to_tensor([img / 255], dtype=tf.float32)

        # predict
        pred = model(img)
        print(pred)
        idx = tf.math.argmax(pred['result'], axis=1).numpy().tolist()
        row[j].write(f'{class_names[idx[0]]} ({pred['result'][0][idx[0]] * 100}%)')
