import os
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

"""
DOCS:
    - P:\cbir\DOCUMENTS\documents\local_features\SIFT\SIFT_applied_to_CBIR.pdf
    - *Main idea: P:\cbir\DOCUMENTS\documents\local_features\SIFT\Content_based_image_retrieval_system_usi.pdf
    - P:\cbir\DOCUMENTS\documents\local_features\SIFT\distinctive_image_features_from_scale-invariant_keypoints.pdf

    - https://docs.opencv.org/4.x/da/df5/tutorial_py_sift_intro.html
    - Matching points between 2 images: https://www.analyticsvidhya.com/blog/2019/10/detailed-guide-powerful-sift-technique-image-matching-python/#:~:text=SIFT%2C%20or%20Scale%20Invariant%20Feature,'keypoints'%20of%20the%20image.
    - https://viblo.asia/p/gioi-thieu-ve-scale-invariant-feature-transform-z3NVRkoLR9xn

    - https://www.quora.com/What-distance-measure-works-best-for-matching-SIFT-feature-descriptors
"""


image_path_1 = "P:/cbir/DATA/wang/image.orig/501.jpg"
# image_path_2 = "P:/cbir/DATA/wang/image.orig/505.jpg"
# image_path_1 = "P:/cbir/DATA/corel/CorelDB/art_1/193001.jpg"
# image_path_2 = "P:/cbir/DATA/corel/CorelDB/art_1/193005.jpg"

image_1 = cv2.imread(image_path_1)
# image_2 = cv2.imread(image_path_2)
# image_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2RGB)
# image_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2RGB)
# image_1 = cv2.resize(image_1, (640, 640))
# image_2 = cv2.resize(image_2, (640, 640))
# gray_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)
# gray_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT_create()

key_points_1 = sift.detect(image_1, None)
# key_points_2 = sift.detect(gray_2, None)


# kp_1, des_1 = sift.compute(gray_1, key_points_1)
# kp_2, des_2 = sift.compute(gray_2, key_points_2)


# bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

# matches = bf.match(des_1, des_2)
# matches = sorted(matches, key = lambda x:x.distance)

# img3 = cv2.drawMatches(image_1, kp_1, image_2, kp_2, matches[:100], gray_2, flags=2)
# plt.imshow(img3),plt.show()
