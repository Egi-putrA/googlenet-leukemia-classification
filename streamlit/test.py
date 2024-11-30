import os

import tensorflow as tf
import cv2
import numpy as np

loaded = tf.saved_model.load('googlenet_model/original_data')

print(list(loaded.signatures.keys()))

infer = loaded.signatures["serving_default"]
print(infer.structured_outputs)

class_names = ['Early', 'Benign', 'Pre', 'Pro']
y_pred = []
y_test = []

for idx, clas in enumerate(class_names):
    clas_folder = os.path.join('data_test', clas)
    for image_name in os.listdir(clas_folder):

        image = cv2.imread(os.path.join(clas_folder, image_name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224)) 

        tensor = tf.convert_to_tensor([image / 255], dtype=tf.float32)

        pred = infer(tensor)
        pred = tf.math.argmax(pred['result'], axis=1).numpy()

        y_test.append(idx)
        y_pred.append(pred[0])

print(tf.math.confusion_matrix(y_test, y_pred).numpy())
