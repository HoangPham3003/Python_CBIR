import os
import cv2
import math
import argparse
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pickle

from tqdm import tqdm


"""
- DOCS:
    - "P:/cbir/DOCUMENTS/documents/Descriptors_of_Mpeg7/Wavelet_features_CLD_EHD_of_Mpeg7/41800235.pdf"
    - "P:/cbir/DOCUMENTS/documents/Descriptors_of_Mpeg7/CLD_EHD.pdf"
    - "P:/cbir/DOCUMENTS/documents/Descriptors_of_Mpeg7/Color_and_Texture_Descriptors.pdf"

- CODE:
    - DCT:
        + https://www.youtube.com/watch?v=mUKPy3r0TTI&ab_channel=ExploringTechnologies
        + https://www.geeksforgeeks.org/discrete-cosine-transform-algorithm-program/
        + https://users.cs.cf.ac.uk/Dave.Marshall/Multimedia/node231.html
        + https://en.wikipedia.org/wiki/Discrete_cosine_transform#Multidimensional_DCTs

    - CLD:
        + https://www.youtube.com/watch?v=lVCRJIPWwu0&ab_channel=ExploringTechnologies
"""


parser = argparse.ArgumentParser()
parser.add_argument('--store_database', type=bool, default=False, help="store database or search image")
parser.add_argument('--query_image_path', type=str, default='', help="store database or search image")
args = parser.parse_args()


class CLDExtractor:
    def __init__(self):
        pass

    # Function to find discrete cosine transform and print it
    def compute_DC_coef(self, block):
        """
        block: (8x8)
        """
        pi = np.pi
        m, n = block.shape

        # dct will store the discrete cosine transform
        # dct = []
        # for i in range(m):
        #     dct.append([None for _ in range(n)])
        
        dct00 = 0
        ci = 1 / (m ** 0.5)
        cj = 1 / (n ** 0.5)

        sum = 0
        for k in range(m):
            for l in range(n):
                dct1 = block[k][l] * math.cos(((2 * k + 1) * 0 * pi) / (2 * m)) * math.cos(((2 * l + 1) * 0 * pi) / (2 * n))
                sum = sum + dct1
        dct00 = ci * cj * sum

        """
        for i in range(m):
            for j in range(n):
    
                # ci and cj depends on frequency as well as
                # number of row and columns of specified matrix
                if (i == 0):
                    ci = 1 / (m ** 0.5)
                else:
                    ci = (2 / m) ** 0.5
                if (j == 0):
                    cj = 1 / (n ** 0.5)
                else:
                    cj = (2 / n) ** 0.5
    
                # sum will temporarily store the sum of
                # cosine signals
                sum = 0
                for k in tqdm(range(m)):
                    for l in range(n):
                        dct1 = block[k][l] * math.cos(((2 * k + 1) * i * pi) / (2 * m)) * math.cos(((2 * l + 1) * j * pi) / (2 * n))
                        sum = sum + dct1
                dct[i][j] = ci * cj * sum
        """
        return dct00


    def compute_dct(self, matrix):
        """
        martix: (m x n) = (256 x 284) (Wang)
        """
        dct = []
        m, n = matrix.shape
        step_y = int(m / 8) 
        step_x = int(n / 8)
        y = 0
        for _ in range(8):
            x = 0
            for _ in range(8):
                block = matrix[y:y+step_y, x:x+step_x]
                DCcoef = self.compute_DC_coef(block=block)
                dct.append(DCcoef)
                x += step_x
            y += step_y
        dct = np.array(dct)
        return dct


    def compute_average(self, matrix):
        h, w = matrix.shape

        M = int(8 * np.ceil(h/8)) 
        N = int(8 * np.ceil(w/8))
        matrix = np.reshape(matrix,(M, N)) # Making image dim. divisible completely by 8

        mask = np.zeros((M, N)) # defining mask to capture only DC coeff. Of DCT
        mask[0, 0] = 1

        matrix_avg = np.ones((M, N))
        step_y = int(M / 8)
        step_x = int(N / 8)
        y = 0
        for _ in range(8):
            x = 0
            for _ in range(8):
                block = matrix[y:y+step_y, x:x+step_x]
                mean_block = np.round(np.mean(block))
                matrix_avg[y:y+step_y, x:x+step_x] = mean_block
                x += step_x
            y += step_y
        return matrix_avg


    def extract_features(self, image_path):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb) # convert to YCrCb
        image = cv2.resize(image, (64, 64))

        avg_Y = self.compute_average(image[:,:,0]) # (256 x 284)
        avg_Cr = self.compute_average(image[:,:,1])
        avg_Cb = self.compute_average(image[:,:,2])

        dct_Y = self.compute_dct(avg_Y)
        dct_Cr = self.compute_dct(avg_Cr)
        dct_Cb = self.compute_dct(avg_Cb)
        features = [dct_Y, dct_Cr, dct_Cb]
        features = np.hstack(features)
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
        
        pickle.dump(features_db, open("./database/features_YCrCb_CorelDataset_resize.pkl", 'wb'))
        pickle.dump(paths_db, open("./database/paths_YCrCb_CorelDataset_resize.pkl", 'wb'))
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
    # DATA_FOLDER = 'P:/cbir/DATA/wang/image.orig/'
    DATA_FOLDER = 'P:/cbir/DATA/corel/CorelDB/'

    create_db = args.store_database
    query_image_path = args.query_image_path
    
    runner = CLDExtractor()
    if create_db:
        """
        Store database
        """
        runner.store_database(data_folder=DATA_FOLDER)
    else:
        """
        Search image
        """
        features_db_file = "./database/features_YCrCb_CorelDataset_resize.pkl"
        file_path_db_file = "./database/paths_YCrCb_CorelDataset_resize.pkl"
        nearest_images = runner.search_image(query_image_path=query_image_path, features_db=features_db_file, paths_db=file_path_db_file)