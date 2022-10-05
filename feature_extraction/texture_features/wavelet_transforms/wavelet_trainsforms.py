import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import mahotas
import pywt


print(pywt.families())

# image_path = "P:/cbir/DATA/wang/image.orig/344.jpg"

# image = cv2.imread(image_path)
# image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)