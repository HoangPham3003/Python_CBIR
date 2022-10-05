import os
import cv2
import numpy as np
import mahotas
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm

"""
DOCS:
    - Function: mahotas.features.zernike_moments(...):
        + https://mahotas.readthedocs.io/en/latest/api.html#mahotas.features.zernike_moments
        + https://github.com/luispedro/mahotas/blob/master/mahotas/features/zernike.py
    - "P:/cbir/DOCUMENTS/documents/features_shape/Zernike_moment/Image Zernike moments shape feature evaluation based on image reconstruction.pdf"
    - "P:/cbir/DOCUMENTS/documents/features_color/color_moment/moment_based_features_for_cbir.pdf"
"""


def extract_features(image_path):
    # loading image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # filtering image
    image = image.max(2)
    
    # radius
    radius = 10
    degree = 50
    # computing zernike moments
    features = mahotas.features.zernike_moments(image, radius=radius, degree=degree)
    return features


def store_database(data_folder):
    features_db = []
    file_path_db = []
    for file_name in tqdm(os.listdir(data_folder)):
        image_path = data_folder + file_name 
        features = extract_features(image_path)
        features_db.append(features)
        file_path_db.append(image_path)
    pickle.dump(features_db, open("./database/features_YCrCb_WangDataset.pkl", 'wb'))
    pickle.dump(file_path_db, open("./database/paths_YCrCb_WangDataset.pkl", 'wb'))
    print("STORE DATABASE SUCCESSFULLY!")


def search_image(features_vetor_query, file_path_query, features_db_file, file_path_db_file):
    features_db = pickle.load(open(features_db_file, 'rb'))
    file_path_db = pickle.load(open(file_path_db_file, 'rb'))
    distances = np.linalg.norm(features_db - features_vetor_query, axis=1)
    
    K = 50
    indices = np.argsort(distances)[:K-1]
    nearest_images = [(features_db[i], file_path_db[i], distances[i]) for i in indices]

    # grid_size = int(math.sqrt(K))
    grid_row = 5
    grid_col = 10
    fig, axes = plt.subplots(grid_row, grid_col, figsize=(12, 6))
    k = 0
    for i in range(grid_row):
        for j in range(grid_col):
            if i == 0 and j == 0:
                axes[i, j].set_title("Query")
                image = cv2.imread(file_path_query)
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
    return nearest_images


if __name__ == '__main__':
    DATA_FOLDER = 'P:/cbir/DATA/wang/image.orig/'
    create_db = False
    if create_db:
        """
        Store database
        """
        store_database(DATA_FOLDER)
    else:
        """
        Search image
        """
        features_db_file = "./database/features_db.pkl"
        file_path_db_file = "./database/file_path_db.pkl"
        features_db_1 = pickle.load(open(features_db_file, 'rb'))
        file_path_db_1 = pickle.load(open(file_path_db_file, 'rb'))
        index_query = 350
        features_vetor_query = features_db_1[index_query]
        file_path_query = file_path_db_1[index_query]
        nearest_images = search_image(features_vetor_query, file_path_query, features_db_file, file_path_db_file)
