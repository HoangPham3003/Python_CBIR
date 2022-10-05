import os
import random
import pandas as pd
import numpy as np

DATA_FOLDER = 'P:/cbir/DATA/corel/CorelDB/'

anchors = []
negatives = []
positives = []

for sub_folder in os.listdir(DATA_FOLDER):
    n_sub_folder = len(os.listdir(DATA_FOLDER))
    sub_folder_path = DATA_FOLDER + sub_folder + '/'
    for file_name in os.listdir(sub_folder_path):
        n_positive = len(os.listdir(sub_folder))
        anchor_file_name = file_name
        for i in range(2):
            # random positive
            index_positive = random.randint(n_positive)
            positive_file_name = os.listdir(sub_folder_path)[index_positive]
            while anchor_file_name == positive_file_name:
                index_positive = random.randint(n_positive)
                positive_file_name = os.listdir(sub_folder_path)[index_positive]

            # random negative
            index_sub_folder = random.randint(n_sub_folder)
            negative_sub_folder = os.listdir(DATA_FOLDER)[index_sub_folder]
            while sub_folder == negative_sub_folder:
                index_sub_folder = random.randint(n_sub_folder)
                negative_sub_folder = os.listdir(DATA_FOLDER)[index_sub_folder]
            negative_sub_folder_path = DATA_FOLDER + negative_sub_folder + '/'
            n_negative = len(os.listdir(negative_sub_folder_path))
            index_negative = random.randint(n_negative)
            negative_file_name = os.listdir(negative_sub_folder_path)[index_negative]

            anchors.append(anchor_file_name)
            negatives.append(negative_file_name)
            positives.append(positive_file_name)

data = {
    'Anchor': anchors,
    'Negative': negatives,
    'Positive': positives
}

df = pd.DataFrame(data=data)
df.to_csv('data.csv')

