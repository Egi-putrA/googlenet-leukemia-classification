import math

import streamlit as st
import tensorflow as tf

class_names = ['Early', 'Benign', 'Pre', 'Pro']

@st.cache_resource
def load_model():
    return tf.saved_model.load('googlenet_model/original_data').signatures["serving_default"]

model = load_model()

images_uploaded = st.file_uploader("Upload file", type=['jpg', 'png'], accept_multiple_files=True)

for i in range(math.ceil(len(images_uploaded) / 3)):
    cols = min(3, len(images_uploaded) - (i*3))
    row = st.columns(cols)

    for j in range(cols):
        row[j].image(images_uploaded[i*3 + j], width=224)
        img = tf.io.decode_image(images_uploaded[i*3 + j].read(), dtype="uint8", channels=3)
        img = tf.image.resize(img, [224, 224])
        img = tf.convert_to_tensor([img / 255], dtype=tf.float32)


        pred = model(img)
        pred = tf.math.argmax(pred['result'], axis=1).numpy().tolist()
        row[j].write(class_names[pred[0]])
