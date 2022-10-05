import os
import cv2
import numpy as np
import pickle
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm

from color_features.color_moments.color_moments import ColorMomentsExtractor
from color_features.color_layout_descriptor.cld import CLDExtractor
from texture_features.edge_histogram_descriptor.ehd import EHDExtractor

# COLOR_HISTOGRAM_DATABASE = "./color_features/color_histogram/database/features_YCrCb_CorelDataset.pkl"
# COLOR_CORRELOGRAM_DATABASE = "./color_features/auto_color_correlogram/database/features_YCrCb_CorelDataset.pkl"
COLOR_MOMENTS_DATABASE = "./color_features/color_moments/database/features_YCrCb_CorelDataset.pkl"
COLOR_CLD_DATABASE = "./color_features/color_layout_descriptor/database/features_YCrCb_CorelDataset_resize.pkl"
TEXTURE_EHD_DATABASE = "./texture_features/edge_histogram_descriptor/database/features_Gray_CorelDataset.pkl"
# TEXTURE_DATABASE = "./texture_features/gabor_fillter/database/features_Ycrcb_CorelDataset.pkl"
PATH_DATABASE = "./color_features/color_moments/database/paths_YCrCb_CorelDataset.pkl"

FEATURES_COMBINATION_DATABASE = "./a_database/features_CorelDataset.pkl"
PATHS_COMBINATION_DATABASE = "./a_database/paths_CorelDataset.pkl"

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--store_database', type=str, default=False, help="store database or search image")
parser.add_argument('--query_image_path', type=str, default='', help="index of query image in database")
args = parser.parse_args()


def store_database():
    # color_histogram_features_db = pickle.load(open(COLOR_HISTOGRAM_DATABASE, 'rb'))
    # color_correlogram_features_db = pickle.load(open(COLOR_CORRELOGRAM_DATABASE, 'rb'))
    color_moments_features_db = pickle.load(open(COLOR_MOMENTS_DATABASE, 'rb'))
    color_cld_features_db = pickle.load(open(COLOR_CLD_DATABASE, 'rb'))
    texture_ehd_features_db = pickle.load(open(TEXTURE_EHD_DATABASE, 'rb'))
    # texture_features_db = pickle.load(open(TEXTURE_DATABASE, 'rb'))
    file_path_db = pickle.load(open(PATH_DATABASE, 'rb'))

    features_database = []
    paths_database = []

    for i in tqdm(range(len(file_path_db))):
        # color_histogram = color_histogram_features_db[i]
        # color_correlogram = color_correlogram_features_db[i]
        color_moments = color_moments_features_db[i]
        color_cld = color_cld_features_db[i]
        texture_ehd = texture_ehd_features_db[i]
        # texture_features = texture_features_db[i]
        features = np.concatenate((color_moments, color_cld, texture_ehd), axis=0)
        features_database.append(features)
        image_path = file_path_db[i]
        paths_database.append(image_path)

    # features_database /= np.linalg.norm(features_database)
    pickle.dump(features_database, open(FEATURES_COMBINATION_DATABASE, 'wb'))
    pickle.dump(paths_database, open(PATHS_COMBINATION_DATABASE, 'wb'))
    print("STORE DATABASE SUCCESSFULLY!")


def search_image(query_image_path, features_db, paths_db):
    features_db = pickle.load(open(features_db, 'rb'))
    paths_db = pickle.load(open(paths_db, 'rb'))

    color_moments_extractor = ColorMomentsExtractor()
    color_cld_extractor = CLDExtractor()
    texture_ehd_extractor = EHDExtractor()

    features_color_moments = color_moments_extractor.extract_features(query_image_path)
    features_color_cld = color_cld_extractor.extract_features(query_image_path)
    features_texture_ehd = texture_ehd_extractor.extract_ehd(query_image_path)

    query_image_features = np.concatenate([features_color_moments, features_color_cld, features_texture_ehd], axis=0)

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


if __name__ == "__main__":
    # DATA_FOLDER = 'P:/cbir/DATA/wang/image.orig/'
    # DATA_FOLDER = 'P:/cbir/DATA/corel/CorelDB/'

    create_db = args.store_database
    query_image_path = args.query_image_path
    
    if create_db:
        """
        Store database
        """
        store_database()
    else:
        """
        Search image
        """
        nearest_images = search_image(query_image_path=query_image_path, features_db=FEATURES_COMBINATION_DATABASE, paths_db=PATHS_COMBINATION_DATABASE)

