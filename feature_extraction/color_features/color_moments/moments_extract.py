import math
import os
import pickle
import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


class ColorMomentsExtractor:
    def __init__(self, image_paths) -> None:
        self.image_paths = image_paths

    def calculate_mean(self, a):
        return np.mean(a)
    
    def calculate_std(self, a):
        mean_a = self.calculate_mean(a)
        return ((1/len(a)) * (sum((a - mean_a) ** 2))) ** (1/2)

    def calculate_skewness(self, a):
        mean_a = self.calculate_mean(a)
        s = (1/len(a)) * (sum((a - mean_a) ** 3))
        if s > 0:
            s = s ** (1/3)
        else:
            s = (abs(s) ** (1/3)) * (-1)
        return s


    def calculate_moments(self, image):
        H = image[:, :, 0].flatten()
        S = image[:, :, 1].flatten()
        V = image[:, :, 2].flatten()

        mean_h = self.calculate_mean(H)
        mean_s = self.calculate_mean(S)
        mean_v = self.calculate_mean(V)

        std_h = self.calculate_std(H)
        std_s = self.calculate_std(S)
        std_v = self.calculate_std(V)

        skewness_h = self.calculate_skewness(H)
        skewness_s = self.calculate_skewness(S)
        skewness_v = self.calculate_skewness(V) 

        mean = [mean_h, mean_s, mean_v]
        std = [std_h, std_s, std_v]
        skewness = [skewness_h, skewness_s, skewness_v]

        features = [mean, std, skewness]
        features = features / np.linalg.norm(features)
        return features

    def extract(self):
        features_list = []
        for image_path in tqdm(self.image_paths):
            image = cv2.imread(image_path) # default flag is BGR
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) # HSV
            image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb) # YCrCb
            image = np.array(image, dtype='float64')
            features = self.calculate_moments(image)
            features_list.append(features)
        return features_list
