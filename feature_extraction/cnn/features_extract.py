import os
import numpy as np
import cv2
from PIL import Image
import torch
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from keras.applications.densenet import preprocess_input

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class ExtractModel:
    def __init__(self):
        self.model = self.ModelCreator() 

    def ModelCreator(self):
        vgg16_model = VGG16(weights="imagenet")
        extract_model = Model(inputs=vgg16_model.inputs, outputs=vgg16_model.get_layer("fc1").output)
        return extract_model


class FeatureExtractor:
    def __init__(self, image_paths: list):
        self.image_paths = image_paths
        self.extract_model = ExtractModel().model

    def preprocessing(self, img):
        img = img.resize((224, 224))
        img = img.convert('RGB')
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        return x

    def feature_extracting(self):
        features = []
        for image_path in self.image_paths:
            img = Image.open(image_path)
            img_tensor = self.preprocessing(img)
            features_vector = self.extract_model.predict(img_tensor)[0]
            features_vector = features_vector / np.linalg.norm(features_vector)
            features.append(features_vector)
            print(image_path)
        return features