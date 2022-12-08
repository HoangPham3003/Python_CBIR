import os
import numpy as np
import cv2
import matplotlib.pyplot as plt


# for idx_orientation in range(n_thetas):
#     theta = idx_orientation * step
#     for sigma in sigmas:
#         for lambd in lambds:
#             for gamma in gammas:
#                 kernel = cv2.getGaborKernel(ksize=ksize, sigma=sigma, theta=theta, lambd=lambd, gamma=gamma, psi=psi)
#                 bank_filters.append(kernel)


# index = 0
# for i in range(4):
#     for j in range(8):
#         filter_image = bank_filters[index]
#         axes.append(fig.add_subplot(4, 8, index + 1))
#         # axes[-1].set_title(filter_image)
#         plt.imshow(filter_image, cmap='gray')
#         plt.axis('off')
#         index += 1
# fig.tight_layout()
# plt.show()

# image_path = "P:/data_test/1.jpg"
# # image_path = "P:/cbir/DATA/corel/CorelDB/pl_flower/84007.jpg"
# img = cv2.imread(image_path)
# # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
# img = img[:,:,0]

# axes = []
# fig = plt.figure(figsize=(12, 6))

# print(bank_filters[0])

# index = 0
# for i in range(4):
#     for j in range(8):
#         kernel = bank_filters[index]
#         res = cv2.filter2D(src=img, ddepth=-1, kernel=kernel)
#         axes.append(fig.add_subplot(4, 8, index + 1))
#         plt.imshow(res, cmap='gray')
#         plt.axis('off')
#         index += 1
# fig.tight_layout()
# plt.show()



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
