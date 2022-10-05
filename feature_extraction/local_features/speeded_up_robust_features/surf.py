import os
import cv2
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt

from mahotas.features.surf import surf


image_path = "P:/cbir/DATA/wang/image.orig/510.jpg"
# image_path_1 = "P:/cbir/DATA/corel/CorelDB/art_1/193001.jpg"


image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

s = cv2.SURF()
kp = s.detect(image, None)
print(kp)
# descriptors = surf(image, descriptor_only=True)
# print(descriptors.shape)