import os
import time
import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm


"""
DOCUMENTS:
- The Gabor filter extract  magnitude at different orientation and scales. 
    For example a magnitude at 0 degree orientation or at 90 degree orientation (or different scales)
    https://stackoverflow.com/questions/20608458/gabor-feature-extraction ****IMPORTANT****
    https://www.researchgate.net/post/What-features-are-extracted-from-Gabor-filter/1
    https://stackoverflow.com/questions/30071474/opencv-getgaborkernel-parameters-for-filter-bank
    https://minhng.info/tutorials/gabor-filters-opencv.html
    https://medium.com/@anuj_shah/through-the-eyes-of-gabor-filter-17d1fdb3ac97
    https://www.youtube.com/watch?v=BTbIS1mriuY&ab_channel=DigitalSreeni
"""


parser = argparse.ArgumentParser()
parser.add_argument('--store_database', type=bool, default=False, help="store database or search image")
parser.add_argument('--image_query_path', type=str, default='', help="store database or search image")
args = parser.parse_args()


def scale_to_0_255(img):
    min_val = np.min(img)
    max_val = np.max(img)
    new_img = (img - min_val) / (max_val - min_val) # 0-1
    new_img *= 255
    return new_img


def apply_sliding_window_on_3_channels(img, kernel):
    """
    CONVOLUTIONAL FUNCTION
    """
    # https://docs.opencv.org/4.4.0/d4/d86/group__imgproc__filter.html#ga27c049795ce870216ddfb366086b5a04
    layer_red = cv2.filter2D(src=img[:,:,0], ddepth=-1, kernel=kernel)
    layer_green = cv2.filter2D(src=img[:,:,1], ddepth=-1, kernel=kernel)
    layer_blue = cv2.filter2D(src=img[:,:,2], ddepth=-1, kernel=kernel)    
    
    new_img = np.zeros(list(layer_blue.shape) + [3])
    new_img[:,:,0], new_img[:,:,1], new_img[:,:,2] = layer_red, layer_green, layer_blue
    return new_img


def generate_gabor_bank(n_thetas=8):
    """
    DOCS:
        - # https://docs.opencv.org/4.4.0/d4/d86/group__imgproc__filter.html#gae84c92d248183bd92fa713ce51cc3599
    CREATE GABOR FILTERS
    Parameters:
        - ksize: Size of the filter returned.
        - sigma: Standard deviation of the gaussian envelope.
        - theta: Orientation of the normal to the parallel stripes of a Gabor function.
        - lambda: Wavelength of the sinusoidal factor.
        - gamma: Spatial aspect ratio.
        - psi: Phase offset.
        - ktype: Type of filter coefficients. It can be CV_32F or CV_64F.
    """
    ksize=(15, 15)
    sigmas=[2, 3] # Brightness
    gammas=[0.25, 0.5] # Blur degree
    lambds=[6] 
    psi=0

    bank_filters = []
    theta = 0
    step = np.pi / n_thetas
    for idx_orientation in range(n_thetas):
        theta = idx_orientation * step
        for sigma in sigmas:
            for lambd in lambds:
                for gamma in gammas:
                    kernel = cv2.getGaborKernel(ksize=ksize, sigma=sigma, theta=theta, lambd=lambd, gamma=gamma, psi=psi)
                    bank_filters.append(kernel)
    return bank_filters


def extract_features(image_path):
    img = cv2.imread(image_path)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    gabor_bank = generate_gabor_bank(n_thetas=8)
    
    # h, w, c = img.shape
    # final_out = np.zeros([h, w*(len(gabor_bank)+1), c])
    # final_out[:, :w, :] = img
    
    avg_out = np.zeros(img.shape)
    

    # local_energy_list = []
    # mean_amplitude_list = []

    mean_list = []
    std_list = []

    for idx, kernel in enumerate(gabor_bank):
        res = apply_sliding_window_on_3_channels(img, kernel)
        # final_out[:, (idx+1)*w:(idx+2)*w, :] = res
        # kh, kw = kernel.shape[:2]
        # kernel_vis = scale_to_0_255(kernel)
        # https://numpy.org/doc/stable/reference/generated/numpy.expand_dims.html
        # https://numpy.org/doc/stable/reference/generated/numpy.repeat.html
        # final_out[:kh, (idx+1)*w:(idx+1)*w+kw, :] = np.repeat(np.expand_dims(kernel_vis, axis=2), repeats=3, axis=2)        
        
        """
        METHOD 1 FOR FEATURE EXTRACTION
        """
        # local_energy = np.sum(res ** 2)
        # mean_amplitude = np.sum(np.abs(res))
        # local_energy_list.append(local_energy)
        # mean_amplitude_list.append(mean_amplitude)

        """
        METHOD 2 FOR FEATURE EXTRACTION
        """
        h, w, c = res.shape
        abs_res = np.abs(res)
        E = np.sum(abs_res)
        mean = E / (h * w * c)
        std = (np.sqrt(np.sum((abs_res - mean) ** 2))) / (h * w * c)
        mean_list.append(mean)
        std_list.append(std)

        # avg_out += res

    # local_energy_list /= np.linalg.norm(local_energy_list)
    # mean_amplitude_list /= np.linalg.norm(mean_amplitude_list)
    # features = np.concatenate([local_energy_list, mean_amplitude_list], axis=0)
    features = np.concatenate([mean_list, std_list], axis=0)
    # avg_out = avg_out / len(gabor_bank)
    # avg_out = avg_out.astype(np.uint8)
    # # cv2.imwrite("result_gabor.jpg", final_out)
    # avg_out = cv2.cvtColor(avg_out, cv2.COLOR_BGR2RGB)
    # cv2.imwrite("result_gabor_avg.jpg", avg_out)
    # print("SAVED!")
    return features       


def store_database(data_folder):
    features_db = []
    file_path_db = []
    """
    WANG DATASET
    """
    # for file_name in tqdm(os.listdir(data_folder)):
    #     image_path = data_folder + file_name 
    #     features = extract_features(image_path)
    #     features_db.append(features)
    #     file_path_db.append(image_path)


    """
    COREL DATASET
    """
    for sub_folder in tqdm(os.listdir(data_folder)):
        sub_folder_path = data_folder + sub_folder + '/'
        for file_name in os.listdir(sub_folder_path):
            if file_name.endswith('.jpg'):
                image_path = sub_folder_path + file_name
                features = extract_features(image_path)
                features_db.append(features)
                file_path_db.append(image_path)
    
    pickle.dump(features_db, open("./database/features_Ycrcb_CorelDataset.pkl", 'wb'))
    pickle.dump(file_path_db, open("./database/paths_Ycrcb_CorelDataset.pkl", 'wb'))
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
    # DATA_FOLDER = 'P:/cbir/DATA/wang/image.orig/'
    DATA_FOLDER = 'P:/cbir/DATA/corel/CorelDB/'
    create_db = args.store_database
    if create_db:
        """
        Store database
        """
        store_database(DATA_FOLDER)
    else:
        """
        Search image
        """
        features_db_file = "./database/features_Ycrcb_CorelDataset.pkl"
        file_path_db_file = "./database/paths_Ycrcb_CorelDataset.pkl"

        file_path_query = args.image_query_path
        features_vetor_query = extract_features(file_path_query)
        nearest_images = search_image(features_vetor_query, file_path_query, features_db_file, file_path_db_file)


    # image_path = "P:/data_test/sudoku.jpg"
    # image_path = "P:/cbir/CBIR_WORKSPACE/data/art_antiques/435016.jpg"
    # start = time.time()
    # features  = extract_features(image_path)
    # print(features.shape)
    # end = time.time()
    # elapsed_time = end - start
    # print('Elapsed time: %.2f second' % elapsed_time)
    # print('---------')