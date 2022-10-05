import os
from features_extract import FeatureExtractor
import pickle

if __name__ == '__main__':
    data_folder = "D:/cbir/CBIR_WorkSpace/DATA/corel/CorelDB/"
    image_path_list = []
    for category_folder in os.listdir(data_folder):
        category_path_folder = data_folder + category_folder + "/"
        for file_name in os.listdir(category_path_folder):
            if file_name.endswith('.jpg'):
                image_path = category_path_folder + file_name
                image_path_list.append(image_path)
    extractor = FeatureExtractor(image_path_list)
    features_vectors = extractor.feature_extracting()


    pickle.dump(image_path_list, open("database/image_paths.pkl", 'wb'))
    pickle.dump(features_vectors, open("database/features_database.pkl", 'wb'))
