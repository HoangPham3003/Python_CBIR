import os
import cv2
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.cluster import KMeans


parser = argparse.ArgumentParser()
parser.add_argument('--store_database', type=bool, default=False, help="store database or search image")
parser.add_argument('--query_image_path', type=str, default='', help="store database or search image")
args = parser.parse_args()


class SIFTMethod2Extractor:
    def __init__(self):
        self.sift = cv2.SIFT_create(contrastThreshold=0, edgeThreshold=0)
        self.k_means = KMeans(n_clusters=16)

    
    def compute_descriptors(self, image_part):
        key_points, descriptors = self.sift.detectAndCompute(image_part, None)
        return key_points, descriptors

    
    def extract_features(self, image_path):
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        h, w = gray.shape
        M = int(2 * np.ceil(h/2))
        N = int(2 * np.ceil(w/2))

        gray = np.reshape(gray, (M, N))
        step_y = int(M / 2)
        step_x = int(N / 2)

        features = []

        y = 0
        for _ in range(2):
            x = 0
            for _ in range(2):
                image_part = gray[y:y+step_y, x:x+step_x]
                key_points, descriptors = self.compute_descriptors(image_part)

                v_hat = np.zeros((len(key_points), 8))

                for i in range(len(key_points)):
                    for j in range(8):
                        for k in range(16):
                            v_hat[i, j] += descriptors[i, 8*k+j]
                
                G = np.zeros((16,))
                if len(v_hat) >= 16:
                    self.k_means.fit(v_hat)
                    predict = self.k_means.predict(v_hat)
                    for y_hat in predict:
                        G[y_hat] += 1
                features.append(G)
            x += step_x
        y += step_y
        features = np.array(features)
        features = features.flatten()
        # try:
        #     features = features / np.linalg.norm(features)
        # except:
        #     pass
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
        
        pickle.dump(features_db, open("./database/features_Gray_WangDataset_method_2_divide.pkl", 'wb'))
        pickle.dump(paths_db, open("./database/paths_Gray_WangDataset_method_2_divide.pkl", 'wb'))
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
    
    runner = SIFTMethod2Extractor()
    if create_db:
        """
        Store database
        """
        runner.store_database(data_folder=DATA_FOLDER)
    else:
        """
        Search image
        """
        features_db_file = "./database/features_Gray_WangDataset_method_2_divide.pkl"
        file_path_db_file = "./database/paths_Gray_WangDataset_method_2_divide.pkl"
        nearest_images = runner.search_image(query_image_path=query_image_path, features_db=features_db_file, paths_db=file_path_db_file)




