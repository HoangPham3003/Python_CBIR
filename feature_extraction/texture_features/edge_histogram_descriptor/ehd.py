import os
import cv2
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm


"""
DOCS:
    - https://github.com/jiyangzhao/pyehd/blob/main/pyehd.py
    - https://www.youtube.com/watch?v=Z7NlH4HC0dw&ab_channel=ExploringTechnologies
    - https://www.section.io/engineering-education/how-to-use-edge-histogram-descriptor-to-retrieve-images-in-matlab/

    - "P:/cbir/DOCUMENTS/documents/Descriptors_of_Mpeg7/Ycrcb_Color_histogram_and_EHD.pdf"
    - "P:/cbir/DOCUMENTS/documents/Descriptors_of_Mpeg7/Wavelet_features_CLD_EHD_of_Mpeg7/41800235.pdf"
    - "P:/cbir/DOCUMENTS/documents/Descriptors_of_Mpeg7/CLD_EHD.pdf"
    - "P:/cbir/DOCUMENTS/documents/Descriptors_of_Mpeg7/Color_and_Texture_Descriptors.pdf"
"""


parser = argparse.ArgumentParser()
parser.add_argument('--store_database', type=bool, default=False, help="store database or search image")
parser.add_argument('--query_image_path', type=str, default='', help="store database or search image")
args = parser.parse_args()

class EHDExtractor:
    def __init__(self):
        pass

    # function to get EHD vector
    def extract_ehd(self, image_path):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # convert to gray
        h, w = np.shape(image) # get the shape of image

        M = 4 * np.ceil(h/4) 
        N = 4 * np.ceil(w/4)
        image = np.reshape(image,(int(M),int(N))) # Making image dim. divisible completely by 4
        AllBins = np.zeros((17, 5)) # initializing Bins
        p = 0
        L = 0
        for _ in range(4):
            K = 0
            for _ in range(4):
                block = image[K:K+int(M/4), L:L+int(N/4)] # Extracting (M/4,N/4) block
                AllBins[p,:] = self.getbins(np.double(block)) 
                K = K + int(M/4)
                p = p + 1
            L = L + int(N/4)
        GlobalBin = np.mean(AllBins[:-1, :], axis=0) # getting global Bin
        AllBins[16,:] = GlobalBin
        # ehd = np.reshape(AllBins,[1,85])
        ehd = np.reshape(AllBins, -1)
        return ehd


    # function for getting Bin values for each block
    def getbins(self, image_block):
        M, N = image_block.shape
        M = 2 * np.ceil(M/2)
        N = 2 * np.ceil(N/2)
        # print(M)
        # print(N)
        image_block = np.reshape(image_block,(int(M),int(N))) # Making block dimension divisible by 2
        bins = np.zeros((1,5)) # initialize Bin
        """Operations, define constant"""
        V = np.array([[1,-1],[1,-1]]) # vertical edge operator
        H = np.array([[1,1],[-1,-1]]) # horizontal edge operator
        D45 = np.array([[1.414,0],[0,-1.414]])# diagonal 45 edge operator
        D135 = np.array([[0,1.414],[-1.414,0]]) # diagonal 135 edge operator
        Isot = np.array([[2,-2],[-2,2]]) # isotropic edge operator
        T = 50 # threshold
        
        nobr = int(M/2) # loop limits
        nobc = int(N/2) # loop limits
        L = 0

        """loops of operating"""
        for _ in range(nobc):
            K = 0
            for _ in range(nobr):
                block = image_block[K:K+2, L:L+2] # Extracting 2x2 block
                pv = np.abs(np.sum(block*V)) # apply operators
                ph = np.abs(np.sum(block*H))
                pd45 = np.abs(np.sum(block*D45))
                pd135 = np.abs(np.sum(block*D135))
                pisot = np.abs(np.sum(block*Isot))
                parray = [pv,ph,pd45,pd135,pisot]
                index = np.argmax(parray) # get the index of max value
                value = parray[index] # get the max value
                # print('value: '+str(value))
                if value >= T:
                    bins[0,index]=bins[0,index]+1 # update bins values
                K = K+2
            L = L+2
        # bins = bins / (nobr * nobc)
        return bins


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
                    features = self.extract_ehd(image_path=image_path)
                    features_db.append(features)
                    paths_db.append(image_path)

        # """
        # WANG DATASET
        # """
        # for file_name in tqdm(os.listdir(data_folder)):
        #     if file_name.endswith('.jpg'):
        #         image_path = data_folder + file_name
        #         features = extract_ehd(image_path=image_path)
        #         features_db.append(features)
        #         paths_db.append(image_path)
        
        pickle.dump(features_db, open("./database/features_Gray_CorelDataset.pkl", 'wb'))
        pickle.dump(paths_db, open("./database/paths_Gray_CorelDataset.pkl", 'wb'))
        print("STORE DATABASE SUCCESSFULLY!")


    def search_image(self, query_image_path, features_db, paths_db):
        features_db = pickle.load(open(features_db, 'rb'))
        paths_db = pickle.load(open(paths_db, 'rb'))
        query_image_features = self.extract_ehd(image_path=query_image_path)
        
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
    
    runner = EHDExtractor()

    if create_db:
        """
        Store database
        """
        runner.store_database(data_folder=DATA_FOLDER)
    else:
        """
        Search image
        """
        features_db_file = "./database/features_Gray_CorelDataset.pkl"
        file_path_db_file = "./database/paths_Gray_CorelDataset.pkl"
        nearest_images = runner.search_image(query_image_path=query_image_path, features_db=features_db_file, paths_db=file_path_db_file)