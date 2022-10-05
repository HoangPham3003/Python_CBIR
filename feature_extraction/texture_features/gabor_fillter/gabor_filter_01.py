import os
import numpy as np
import cv2
import matplotlib.pyplot as plt


ksize = 5
sigma = 3
theta = 2 * np.pi / 4
lambd = 1 * np.pi / 4
gamma = 0.5
psi = 0
ktype = cv2.CV_32F

kernel = cv2.getGaborKernel(ksize=(ksize, ksize), sigma=sigma, theta=theta, lambd=lambd, gamma=gamma, psi=psi, ktype=ktype)

image_path = "P:/cbir/CBIR_WORKSPACE/data/art_1/193009.jpg"

img = cv2.imread(image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

fimg = cv2.filter2D(img, cv2.CV_8UC3, kernel)
kernel_resized = cv2.resize(kernel, (400, 400))

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 6))
axes[0].imshow(img, cmap='gray', vmin=0, vmax=255)
axes[1].imshow(kernel)
axes[2].imshow(fimg, cmap='gray', vmin=0, vmax=255)

plt.tight_layout()
plt.show()
