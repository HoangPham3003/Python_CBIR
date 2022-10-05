import os
import math
import numpy as np
from PIL import Image
import pickle
import matplotlib.pyplot as plt
from features_extract import FeatureExtractor


vectors = pickle.load(open("D:/cbir/CBIR_WorkSpace/cnn/database/features_database.pkl", 'rb'))
paths = pickle.load(open("D:/cbir/CBIR_WorkSpace/cnn/database/image_paths.pkl", 'rb'))


if __name__ == '__main__':
    image_paths_list = []
    image_path = "D:/cbir/CBIR_WorkSpace/DATA_INFERENCE/5.jpg"
    image_paths_list.append(image_path)
    extractor = FeatureExtractor(image_paths=image_paths_list)
    search_vector = extractor.feature_extracting()[0]
    distance = np.linalg.norm(vectors - search_vector, axis=1)
    K = 49
    indexs = np.argsort(distance)[:K]

    nearest_images = [(paths[id], distance[id]) for id in indexs]
    print(nearest_images)

    axes = []
    grid_size = int(math.sqrt(K))
    fig = plt.figure(figsize=(15, 8))
    axes.append(fig.add_subplot(grid_size, grid_size, 1))
    axes[-1].set_title("Query image")
    plt.imshow(Image.open(image_path))
    for i in range(K-1):
        draw_image = nearest_images[i]
        axes.append(fig.add_subplot(grid_size, grid_size, i + 2))
        axes[-1].set_title(draw_image[1])
        plt.imshow(Image.open(draw_image[0]))
    fig.tight_layout()
    plt.show()
