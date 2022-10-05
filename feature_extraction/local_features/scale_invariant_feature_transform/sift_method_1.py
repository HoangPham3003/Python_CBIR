import os
import cv2
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from tqdm import tqdm


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


parser = argparse.ArgumentParser()
parser.add_argument('--store_database', type=bool, default=False, help="store database or search image")
parser.add_argument('--query_image_path', type=str, default='', help="store database or search image")
args = parser.parse_args()


class SIFTMethod1Extractor:
    def __init__(self):
        self.sift = cv2.SIFT_create()
        self.k_means = KMeans(n_clusters=8, init="k-means++", n_init=12)


    def compute_descriptors(self, image_path):
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        key_points, descriptors = self.sift.detectAndCompute(gray, None)
        return key_points, descriptors

    
    def extract_features(self, image_path):
        key_points, descriptors = self.compute_descriptors(image_path)
        X = []
        step = 8
        for des in descriptors:
            for i in range(16):
                X.append(des[8*i:8*i+8])
        X = np.array(X)

        self.k_means.fit(X)
        predict = self.k_means.predict(X)

        M = np.zeros((8, 16))
        for i in range(0, len(predict), 16):
            kp = predict[i:i+16] # a key point includes 16 parts
            for j in range(16):
                M[kp[j], j] += 1 # kp[j]: index of cluster (0 -> 7); j: index of part (0 -> 15)
        features = M.flatten()
        return features

    
    def store_database(self, data_folder):
        features_db = []
        paths_db = []

        """
        COREL DATASET
        """
        # for sub_folder in os.listdir(data_folder):
        #     print(f"=====> {sub_folder}")
        #     sub_folder_path = data_folder + sub_folder + '/'
        #     for file_name in tqdm(os.listdir(sub_folder_path)):
        #         if file_name.endswith('.jpg'):
        #             image_path = sub_folder_path + file_name
        #             features = self.extract_features(image_path=image_path)
        #             features_db.append(features)
        #             paths_db.append(image_path)

        """
        WANG DATASET
        """
        for file_name in tqdm(os.listdir(data_folder)):
            if file_name.endswith('.jpg'):
                image_path = data_folder + file_name
                features = self.extract_features(image_path=image_path)
                features_db.append(features)
                paths_db.append(image_path)
        
        pickle.dump(features_db, open("./database/features_Gray_WangDataset_method_1.pkl", 'wb'))
        pickle.dump(paths_db, open("./database/paths_Gray_WangDataset_method_1.pkl", 'wb'))
        print("STORE DATABASE SUCCESSFULLY!")


    def search_image(self, query_image_path, features_db, paths_db):
        features_db = pickle.load(open(features_db, 'rb'))
        paths_db = pickle.load(open(paths_db, 'rb'))
        query_image_features = self.extract_features(image_path=query_image_path)
        
        distances = np.linalg.norm(features_db - query_image_features, axis=1)
        K = 50
        indexs = np.argsort(distances)[:K]

        nearest_images = [(features_db[id], paths_db[id], distances[id]) for id in indexs]

        # grid_size = int(math.sqrt(K))
        grid_row = 5
        grid_col = 10
        fig, axes = plt.subplots(grid_row, grid_col, figsize=(12, 6))
        k = 0
        for i in range(grid_row):
            for j in range(grid_col):
                if i == 0 and j == 0:
                    axes[i, j].set_title("Query")
                    image = cv2.imread(query_image_path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    axes[i, j].imshow(image)
                    axes[i, j].axis('off')
                    k += 1
                else:
                    features_vector, file_path, distance = nearest_images[k-1]
                    # axes[i, j].set_title(distance)
                    image = cv2.imread(file_path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    axes[i, j].imshow(image)
                    axes[i, j].axis('off')
                    k += 1
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    DATA_FOLDER = 'P:/cbir/DATA/wang/image.orig/'
    # DATA_FOLDER = 'P:/cbir/DATA/corel/CorelDB/'

    create_db = args.store_database
    query_image_path = args.query_image_path
    
    runner = SIFTMethod1Extractor()
    if create_db:
        """
        Store database
        """
        runner.store_database(data_folder=DATA_FOLDER)
    else:
        """
        Search image
        """
        features_db_file = "./database/features_Gray_WangDataset_method_1.pkl"
        file_path_db_file = "./database/paths_Gray_WangDataset_method_1.pkl"
        nearest_images = runner.search_image(query_image_path=query_image_path, features_db=features_db_file, paths_db=file_path_db_file)

# image_path = "P:/cbir/DATA/wang/image.orig/501.jpg"
# image_path_2 = "P:/cbir/DATA/wang/image.orig/505.jpg"
# image_path_1 = "P:/cbir/DATA/corel/CorelDB/art_1/193001.jpg"
# image_path_2 = "P:/cbir/DATA/corel/CorelDB/art_1/193005.jpg"


# image_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2RGB)
# image_1 = cv2.resize(image_1, (640, 640))



