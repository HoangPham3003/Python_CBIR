import os
import pickle
import cv2
import matplotlib.pyplot as plt
import numpy as np
import argparse
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--store_database', type=bool, default=False, help="store database or search image")
parser.add_argument('--query_image_path', type=str, default='', help="store database or search image")
args = parser.parse_args()


class ColorMomentsExtractor:
    def __init__(self):
        pass

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
        features = np.array(features)
        features = np.hstack(features)
        # features = features / np.linalg.norm(features)
        return features


    def extract_features(self, image_path):
        image = cv2.imread(image_path) # default flag is BGR
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) # HSV
        image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb) # YCrCb
        image = np.array(image, dtype='float64')
        features = self.calculate_moments(image)
        return features


    def store_database(self, data_folder):
        features_db = []
        paths_db = []

        """
        COREL DATASET
        """
        for sub_folder in os.listdir(data_folder):
            print(f"=====> {sub_folder}")
            sub_folder_path = data_folder + sub_folder + '/'
            for file_name in tqdm(os.listdir(sub_folder_path)):
                if file_name.endswith('.jpg'):
                    image_path = sub_folder_path + file_name
                    features = self.extract_features(image_path=image_path)
                    features_db.append(features)
                    paths_db.append(image_path)

        """
        WANG DATASET
        """
        # for file_name in tqdm(os.listdir(data_folder)):
        #     if file_name.endswith('.jpg'):
        #         image_path = data_folder + file_name
        #         features = extract_features(image_path=image_path)
        #         features_db.append(features)
        #         paths_db.append(image_path)
        
        pickle.dump(features_db, open("./database/features_YCrCb_CorelDataset.pkl", 'wb'))
        pickle.dump(paths_db, open("./database/paths_YCrCb_CorelDataset.pkl", 'wb'))
        print("STORE DATABASE SUCCESSFULLY!")

    
    def calculate_distance(self, features_list, search_features):
        W = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
        distances = []
        for features in features_list:
            distance = 0
            for channel in range(3):
                for moment in range(3):
                    distance += W[moment][channel] * abs(features[moment][channel] - search_features[moment][channel])
            distances.append(distance)
        return distances


    def search_image(self, query_image_path, features_db, paths_db):
        features_list = pickle.load(open(features_db, 'rb'))
        paths_db = pickle.load(open(paths_db, 'rb'))


        search_features = self.extract_features(image_path=query_image_path)
        # distances = self.calculate_distance(features_list=features_list, search_features=search_features)
        # features_list = np.resize(features_list, (len(features_list), 9))
        # print(features_list.shape)
        # search_features = search_features.flatten()
        # print(search_features.shape)
        distances = np.linalg.norm(features_list - search_features, axis=1)

        K = 50
        indices = np.argsort(distances)[:K-1]
        nearest_images = [(features_list[i], paths_db[i], distances[i]) for i in indices]

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
    # DATA_FOLDER = 'P:/cbir/DATA/wang/image.orig/'
    DATA_FOLDER = 'P:/cbir/DATA/corel/CorelDB/'

    create_db = args.store_database
    query_image_path = args.query_image_path
    
    runner = ColorMomentsExtractor()

    if create_db:
        """
        Store database
        """
        runner.store_database(data_folder=DATA_FOLDER)
    else:
        """
        Search image
        """
        features_db_file = "./database/features_YCrCb_CorelDataset.pkl"
        file_path_db_file = "./database/paths_YCrCb_CorelDataset.pkl"
        nearest_images = runner.search_image(query_image_path=query_image_path, features_db=features_db_file, paths_db=file_path_db_file)