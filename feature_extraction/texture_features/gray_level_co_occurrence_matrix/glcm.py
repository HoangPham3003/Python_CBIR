import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import argparse

from skimage.feature import graycomatrix, graycoprops
from tqdm import tqdm

"""
DOCS:
    - "P:/cbir/DOCUMENTS/documents/features_texture/Texture_IJLTC_(#important).pdf"
    - Code example: https://github.com/alfianhid/Feature-Extraction-Gray-Level-Co-occurrence-Matrix-GLCM-with-Python/blob/master/Feature_Extraction_Gray_Level_Co_occurrence_Matrix_(GLCM)_with_Python.ipynb
    - scikit-image: https://scikit-image.org/docs/dev/auto_examples/features_detection/plot_glcm.html
"""


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--store_database', type=bool, default=False, help="store database or search image")
parser.add_argument('--image_query_path', type=str, default='', help="store database or search image")
args = parser.parse_args()

def calculate_entropy(glcm):
    a, b, c, d = glcm.shape # (256, 256, 1, 4)
    entropies = []
    for v in range(d): # 4
        s = 0
        for u in range(c): # 1
            for j in range(b): # 256
                for i in range(a): # 256
                    s += (-1) * glcm[i, j, u, v] * np.log(glcm[i, j, u, v] + 0.00000000001)
        entropies.append(s)
    return entropies
            

# ----------------- calculate greycomatrix() & greycoprops() for angle 0, 45, 90, 135 ----------------------------------
def calc_glcm_all_agls(image_path, props, distances=[1, 2, 5, 10, 20, 50], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], lvl=256, sym=True, norm=False):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # call greycomatrix function from skimage
    glcm = graycomatrix(image=image, 
                        distances=distances, 
                        angles=angles, 
                        levels=lvl,
                        symmetric=sym, 
                        normed=norm)
    
    features = []
    # calculate properties from greycoprops function from skimage
    glcm_props = [property for name in props for property in graycoprops(glcm, name)[0]]
    for item in glcm_props:
        features.append(item)
    # entropy = calculate_entropy(glcm)
    # for x in entropy:
    #     features.append(x)
    features = np.array(features)
    return features


def store_database(data_folder, properties):
    # ----------------- call calc_glcm_all_agls() for all properties ----------------------------------
    features_db = []
    file_path_db = []
    """
    COREL DATASET
    """
    # for sub_folder in tqdm(os.listdir(data_folder)):
    #     sub_folder_path = data_folder + sub_folder + '/'
    #     for file_name in os.listdir(sub_folder_path):
    #         image_path = sub_folder_path + file_name
    #         if image_path.endswith('.jpg'):
    #             glcm_features = calc_glcm_all_agls(image_path, props=properties)
    #             features_db.append(glcm_features)
    #             file_path_db.append(image_path)
    """
    WANG DATASET
    """
    for file_name in tqdm(os.listdir(data_folder)):
        image_path = data_folder + file_name
        glcm_features = calc_glcm_all_agls(image_path, props=properties)
        features_db.append(glcm_features)
        file_path_db.append(image_path)

    pickle.dump(features_db, open("./database/features_WangDataset.pkl", 'wb'))
    pickle.dump(file_path_db, open("./database/paths_WangDataset.pkl", 'wb'))
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


if __name__ == "__main__":
    # properties = ['dissimilarity', 'correlation', 'homogeneity', 'contrast', 'ASM', 'energy']
    properties = ['correlation', 'contrast', 'homogeneity', 'energy']

    DATA_FOLDER = 'P:/cbir/DATA/wang/image.orig/'
    # DATA_FOLDER = "P:/cbir/DATA/corel/CorelDB/"
    create_db = args.store_database
    if create_db:
        """
        Store database
        """
        store_database(DATA_FOLDER, properties)
    else:
        """
        Search image
        """
        features_db_file = "./database/features_WangDataset.pkl"
        file_path_db_file = "./database/paths_WangDataset.pkl"

        file_path_query = args.image_query_path
        
        features_vetor_query = calc_glcm_all_agls(file_path_query, props=properties)
        nearest_images = search_image(features_vetor_query, file_path_query, features_db_file, file_path_db_file)


