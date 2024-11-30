import streamlit as st
import tensorflow as tf
import cv2
import numpy as np

@st.cache_resource
def load_model():
    return tf.saved_model.load('googlenet_model/original_data')

model = load_model()
print(model)

images_uploaded = st.file_uploader("Upload file", type=['jpg', 'png'], accept_multiple_files=True)


for image in images_uploaded:
    print(type(image))
    print(image)
    image_data = np.asarray(bytearray(image.read()), dtype="uint8")
    img_file = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
    cv2.imwrite("result.jpg", img_file) 
    st.image(image)

    result = model.predict(img_file)
    print(result)
    print(tf.math.argmax(result, axis=1))
