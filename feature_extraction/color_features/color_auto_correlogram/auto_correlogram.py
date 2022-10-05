from cmath import isnan
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle
import argparse

from tqdm import tqdm

"""
DOCS:
    - https://medium.com/@shashwat17103/the-dummys-guide-to-colour-correlogram-from-scratch-in-python-1b20a55eb00c
    - https://www.youtube.com/watch?v=L1qqZvrPlyo&t=1181s&ab_channel=IT
    - "P:/cbir/DOCUMENTS/documents/features_color/color_correlogram/color_correlogram_02.pdf"
"""

parser = argparse.ArgumentParser()
parser.add_argument('--store_database', type=bool, default=False, help="store database or search image")
parser.add_argument('--query_image_path', type=str, default='', help="store database or search image")
parser.add_argument('--color_space', type=str, default='YCrCb', help="store database or search image")
args = parser.parse_args()


def get_probdist(img, bins, d):
    colours = []
    for c1 in bins:
        for c2 in bins:
            for c3 in bins:
                colours.append([c1, c2, c3])
    carray = []
    for i in img:
        arr = []
        n = 256 / len(bins)
        for j in i:
            r = (j[0] // n) * n
            g = (j[1] // n) * n
            b = (j[2] // n) * n
            index = colours.index([r, g, b])
            arr.append(index) # bins value of each pixel in each row
        carray.append(arr) # whole image
    carray = np.array(carray)


    a = np.zeros(shape=(64, len(d), 2)) # Shape = (index of bins, index of distance, track), track: 0: same color, 1: different color
    for i in range(0, len(carray)): # loop for each row => carray[i] : a row in image
        for j in range(len(carray[i])): # lopp for each pixel in each row i
            clr = carray[i][j] # bins value of each image
            for z in range(len(d)): # loop for list of distances
                k = d[z]
                for trvlr in range(1, k+1):
                    # calculate chess-board distance
                    try:
                        one = carray[i+trvlr][j+k]
                        if one == clr:
                            a[clr][z][0] = a[clr][z][0] + 1
                            a[clr][z][1] = a[clr][z][1] + 1
                        else:
                            a[clr][z][1] = a[clr][z][1] + 1
                    except:
                        pass

                    try:
                        two = carray[i-trvlr][j+k]
                        if two == clr:
                            a[clr][z][0] = a[clr][z][0] + 1
                            a[clr][z][1] = a[clr][z][1] + 1
                        else:
                            a[clr][z][1] = a[clr][z][1] + 1
                    except:
                        pass

                    try:
                        three = carray[i+trvlr][j-k]
                        if three == clr:
                            a[clr][z][0] = a[clr][z][0] + 1
                            a[clr][z][1] = a[clr][z][1] + 1
                        else:
                            a[clr][z][1] = a[clr][z][1] + 1
                    except:
                        pass

                    try:
                        four = carray[i-trvlr][j-k]
                        if four == clr:
                            a[clr][z][0] = a[clr][z][0] + 1
                            a[clr][z][1] = a[clr][z][1] + 1
                        else:
                            a[clr][z][1] = a[clr][z][1] + 1
                    except:
                        pass

                    try:
                        five = carray[i+k][j-trvlr]
                        if five == clr:
                            a[clr][z][0] = a[clr][z][0] + 1
                            a[clr][z][1] = a[clr][z][1] + 1
                        else:
                            a[clr][z][1] = a[clr][z][1] + 1
                    except:
                        pass

                    try:
                        six = carray[i+k][j+trvlr]
                        if six == clr:
                            a[clr][z][0] = a[clr][z][0] + 1
                            a[clr][z][1] = a[clr][z][1] + 1
                        else:
                            a[clr][z][1] = a[clr][z][1] + 1
                    except:
                        pass

                    try:
                        seven = carray[i-k][j-trvlr]
                        if seven == clr:
                            a[clr][z][0] = a[clr][z][0] + 1
                            a[clr][z][1] = a[clr][z][1] + 1
                        else:
                            a[clr][z][1] = a[clr][z][1] + 1
                    except:
                        pass

                    try:
                        eight = carray[i-k][j+trvlr]
                        if eight == clr:
                            a[clr][z][0] = a[clr][z][0] + 1
                            a[clr][z][1] = a[clr][z][1] + 1
                        else:
                            a[clr][z][1] = a[clr][z][1] + 1
                    except:
                        pass

    colour_dist_prob = []

    for i in a: # loop for list of bins color
        l = []
        for j in i: # loop for list of distances
            div = j[0] / j[1]
            if np.isnan(div):
                l.append(0) 
            else:
                l.append(div)
        colour_dist_prob.append(l)

    correlogram = np.array(colour_dist_prob)
    correlogram = correlogram.flatten()
    return correlogram


def store_database(data_folder, color_space, bins, d):
    """
    - data_folder: folder of dataset (Wang, Corel)
    - color_space: color space (RGB, HSV, YCrcb)
    - bins: bins of color
    - d: distances between pixels
    """

    features_db = []
    paths_db = []
    """
    COREL DATASET
    """
    folders = os.listdir(data_folder)
    # folders.sort()
    for sub_folder in folders:
        sub_path_folder = data_folder + sub_folder + "/"
        print(f"=====> {sub_folder}")
        for file_name in tqdm(os.listdir(sub_path_folder)):
            if file_name.endswith('.jpg'):
                image_path = sub_path_folder + file_name
                image = cv2.imread(image_path)
                if color_space == 'RGB':
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                elif color_space == 'HSV':
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                else:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
                features = get_probdist(img=image, bins=bins, d=d)
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
    
    pickle.dump(features_db, open(f"./features_{color_space}_CorelDataset.pkl", 'wb'))
    pickle.dump(paths_db, open(f"./paths_{color_space}_CorelDataset.pkl", 'wb'))
    print("STORE DATABASE SUCCESSFULLY!")


def search_image(query_image_path, color_space, features_db_path, paths_db_path, bins, d):
    features_db = pickle.load(open(features_db_path, 'rb'))
    paths_db = pickle.load(open(paths_db_path, 'rb'))

    # image_paths = [WORK_DIR + "data/art_dino/644012.jpg"]
    # image_paths = "P:/cbir/DATA/wang/image.orig/741.jpg"
    
    image = cv2.imread(query_image_path)
    if color_space == 'RGB':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif color_space == 'HSV':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

    query_image_features = get_probdist(img=image, bins=bins, d=d)

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
    return nearest_images


if __name__ == '__main__':
    # features_db = pickle.load(open("./database/features_YCrCb_CorelDataset.pkl", 'rb'))
    # paths_db = pickle.load(open("./database/paths_YCrCb_CorelDataset.pkl", 'rb'))
    # paths_db = np.array(paths_db)
    # indices = np.argsort(paths_db)

    # new_features_db = []
    # new_paths_db = []
    # for idx in tqdm(indices):
    #     path = paths_db[idx]
    #     features = features_db[idx]
    #     new_features_db.append(features)
    #     new_features_db.append(path)
    # pickle.dump(features_db, open(f"./database/features_YCrCb_CorelDataset_new.pkl", 'wb'))
    # pickle.dump(paths_db, open(f"./database/paths_YCrCb_CorelDataset_new.pkl", 'wb'))


    # DATA_FOLDER = 'P:/cbir/DATA/wang/image.orig/'
    DATA_FOLDER = 'P:/cbir/DATA/corel/CorelDB/'

    bins = [0, 64, 128, 192]
    d = [1, 3, 5]

    create_db = args.store_database
    query_image_path = args.query_image_path
    color_space = args.color_space
    
    if create_db:
        """
        Store database
        """
        store_database(data_folder=DATA_FOLDER, color_space=color_space, bins=bins, d=d)
    else:
        """
        Search image
        """
        features_db_file = "./database/features_YCrCb_CorelDataset.pkl"
        file_path_db_file = "./database/paths_YCrCb_CorelDataset.pkl"
        nearest_images = search_image(query_image_path, color_space, features_db_file, file_path_db_file, bins, d)
