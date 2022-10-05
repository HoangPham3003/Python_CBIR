import os
from turtle import forward
import cv2
from PIL import Image
import numpy as np
import argparse
import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
from torchvision.models.vgg import VGG16_Weights
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle

# os.environ['KMP_DUPLICATE_LIB_OK']='True'

parser = argparse.ArgumentParser()
parser.add_argument('--store_database', type=bool, default=False, help="store database or search image")
parser.add_argument('--query_image_path', type=str, default='', help="store database or search image")
parser.add_argument('--color_space', type=str, default='YCrCb', help="store database or search image")
args = parser.parse_args()


class VGG16Model(nn.Module):
    def __init__(self):
        super(VGG16Model, self).__init__()
        model = torchvision.models.vgg16(weights=VGG16_Weights.IMAGENET1K_FEATURES)
        self.features = model.features
        self.avgpool = model.avgpool
        self.flatten = nn.Flatten()
    
    def forward(self, x):
        out = self.features(x)
        out = self.avgpool(out)
        out = self.flatten(out)
        return out

def feature_extraction(image_path, model, device, transform):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    image = Image.fromarray(image)
    # image = image.convert('RGB')
    image = transform(image)
    image = image.unsqueeze(0)
    image = image.to(device)
    output = model(image)
    features = output.detach().numpy()
    features = np.reshape(features, -1)
    return features


def store_database(data_folder, device, model, transform):
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
                features = feature_extraction(image_path=image_path, device=device, model=model, transform=transform)
                features_db.append(features)
                paths_db.append(image_path)

    # """
    # WANG DATASET
    # """
    # for file_name in tqdm(os.listdir(data_folder)):
    #     if file_name.endswith('.jpg'):
    #         image_path = data_folder + file_name
    #         features = feature_extraction(image_path=image_path, device=device, model=model, transform=transform)
    #         features_db.append(features)
    #         paths_db.append(image_path)
    
    pickle.dump(features_db, open("./database/features_VGG16_YCrCb_CorelDataset.pkl", 'wb'))
    pickle.dump(paths_db, open("./database/paths_VGG16_YCrCb_CorelDataset.pkl", 'wb'))
    print("STORE DATABASE SUCCESSFULLY!")


def search_image(query_image_path, features_db, paths_db, model, device, transform):
    features_db = pickle.load(open(features_db, 'rb'))
    paths_db = pickle.load(open(paths_db, 'rb'))
    query_image_features = feature_extraction(image_path=query_image_path, model=model, device=device, transform=transform)
    
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vgg16_model = VGG16Model()

    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # DATA_FOLDER = 'P:/cbir/DATA/wang/image.orig/'
    DATA_FOLDER = 'P:/cbir/DATA/corel/CorelDB/'

    create_db = args.store_database
    query_image_path = args.query_image_path
    color_space = args.color_space
    
    if create_db:
        """
        Store database
        """
        store_database(data_folder=DATA_FOLDER, device=device, model=vgg16_model, transform=transform)
    else:
        """
        Search image
        """
        features_db_file = "./database/features_VGG16_YCrCb_CorelDataset.pkl"
        file_path_db_file = "./database/paths_VGG16_YCrCb_CorelDataset.pkl"
        nearest_images = search_image(query_image_path=query_image_path,features_db=features_db_file, paths_db=file_path_db_file, device=device, model=vgg16_model, transform=transform)