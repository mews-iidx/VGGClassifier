from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.preprocessing import image
import numpy as np
import cv2
import os
import tensorflow as tf

class VGGClassifier(object):
    def __init__(self):
        self.sess = tf.Session()
        tf.keras.backend.set_session(self.sess)

        self.model = VGG16(weights='imagenet')
        self.model._make_predict_function()
        self.graph = tf.get_default_graph()

    def check_summary(self):
        return self.model.summary()

    def predict(self, images_as_list_of_arrays, _):
        with self.graph.as_default():
            tf.keras.backend.set_session(self.sess)
            img_arrays_list = []
            for img in images_as_list_of_arrays:
                img_np = np.array(img, dtype=np.uint8)
                target_size = (224, 224)
                img_resized = cv2.resize(img_np, target_size)
                img_arrays_list.append(img_resized)

            npa = np.array(img_arrays_list)
            prep = preprocess_input(npa)
            preds = self.model.predict(prep)
            results = decode_predictions(preds, top=5)
            return results
