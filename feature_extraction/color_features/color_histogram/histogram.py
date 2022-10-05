import os
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
import pickle
import argparse

from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--store_database', type=bool, default=False, help="store database or search image")
parser.add_argument('--query_image_path', type=str, default='', help="store database or search image")
parser.add_argument('--color_space', type=str, default='YCrCb', help="store database or search image")
args = parser.parse_args()


"""
Color histogram:
- Calculate histogram for 3 channel RGB or HSV: hist[0], hist[1], hist[2]
- Summary feature vector = (hist[0] concate hist[1] concat hist[2]) / (heigt * width)
- Note*: do not hist[0] + hist[1] + hist[2]

Range of YCbCr:
    - Y = [0:255]
    - Cb = [0:255]
    - Cr = [0:255]
https://stackoverflow.com/questions/53405148/what-is-the-range-of-pixel-values-in-hsv-ycrcb-and-lab-color-spaces-for-each
"""

def extract_features(image_path, color_space):
    image = cv2.imread(image_path) # default flag is BGR
    image = cv2.resize(image, (224, 224))
    h, w, c = image.shape
    if color_space == 'HSV':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    elif color_space == 'RGB':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

    
    #     features = calculate_histogram(image)
    #     features_vectors.append(features)
    # width, height, dimemsion = image.shape
    
    """
    Histogram for YCrCb
    n_bins_Y: 8
    n_bins_Cr: 4
    n_bins_Cb: 4
    """
    histogram = cv2.calcHist([image], [0, 1, 2], None, [8, 4, 4], [0, 256, 0, 256, 0, 256]) 
    features = histogram.flatten()
    features /= (h * w)
    # features /= np.linalg.norm(features)
    return features


def store_database(data_folder, color_space):
    features_db = []
    paths_db = []
    """
    COREL DATASET
    """
    for sub_folder in tqdm(os.listdir(data_folder)):
        sub_path_folder = data_folder + sub_folder + "/"
        for file_name in os.listdir(sub_path_folder):
            if file_name.endswith('.jpg'):
                image_path = sub_path_folder + file_name
                features = extract_features(image_path=image_path, color_space=color_space)
                features_db.append(features)
                paths_db.append(image_path)
    
    # ===================================================
    """
    WANG DATASET
    """
    # for file_name in tqdm(os.listdir(data_folder)):
    #     if file_name.endswith('.jpg'):
    #         image_path = data_folder + file_name
    #         features = extract_features(image_path=image_path, color_space=color_space)
    #         features_db.append(features)
    #         paths_db.append(image_path)
    
    pickle.dump(features_db, open(f"database/features_{color_space}_CorelDataset_test.pkl", 'wb'))
    pickle.dump(paths_db, open(f"database/paths_{color_space}_CorelDataset_test.pkl", 'wb'))
    print("STORE DATABASE SUCCESSFULLY!")


def search_image(query_image_path, color_space, features_db, paths_db):
    features_db = pickle.load(open(features_db, 'rb'))
    paths_db = pickle.load(open(paths_db, 'rb'))

    # image_paths = [WORK_DIR + "data/art_dino/644012.jpg"]
    # image_paths = "P:/cbir/DATA/wang/image.orig/741.jpg"

    query_image_features = extract_features(image_path=query_image_path, color_space=color_space)

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
    color_space = args.color_space
    
    if create_db:
        """
        Store database
        """
        store_database(data_folder=DATA_FOLDER, color_space=color_space)
    else:
        """
        Search image
        """
        features_db_file = "./database/features_Ycrcb_CorelDataset_resize.pkl"
        file_path_db_file = "./database/paths_Ycrcb_CorelDataset_resize.pkl"
        nearest_images = search_image(query_image_path, color_space, features_db_file, file_path_db_file)
