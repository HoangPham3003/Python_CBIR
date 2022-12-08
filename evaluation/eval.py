import os
import cv2
from PIL import Image
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt

from tensorflow import keras
from keras.applications.vgg19 import VGG19
from keras.models import Model
from keras.utils import img_to_array
from keras.applications.densenet import preprocess_input


from tqdm import tqdm
import pickle

from sklearn import svm


"""
DOCUMENTS RF:
    - #Important: https://www.ee.columbia.edu/ln/dvmm/workshop-IQR/presentations/RF_talk_Hoi.pdf
    - RF using SVM + Active learning: P:\cbir\DOCUMENTS\documents\Relavance_feedback\Image_Retrieval_with_Relevance_Feedback_using_SVM_.pdf
    - https://www.slideshare.net/trongthuy3/luan-van-tra-cuu-anh-dua-tren-noi-dung-su-dung-nhieu-dac-trung-hay
    - http://image.diku.dk/jank/papers/WIREs2014.pdf
    - Rocchio: https://en.wikipedia.org/wiki/Rocchio_algorithm#cite_note-ir-manning-1

Active learning:
    - https://viblo.asia/p/lam-gi-khi-mo-hinh-hoc-may-thieu-du-lieu-co-nhan-phan-1-tong-quan-ve-active-learning-RQqKLRLbl7z
    - https://burrsettles.com/pub/settles.activelearning.pdf


Distance from a data point to hyperplane in SVM:
    - https://stackoverflow.com/questions/32074239/sklearn-getting-distance-of-each-point-from-decision-boundary

"""

"""
Chú ý:
    - Sau mỗi bước thực hiện RF, đã thực hiện phân loại cho k unlabeled samples vào 2 loại pos hoặc neg
    - Rõ ràng rằng các mẫu được phân loại là pos sẽ được chọn vào ground truth positive cho bước RF tiếp theo
    - Tuy nhiên, trong code này, tại bước RF tiếp theo, các mẫu đã được phân loại và không chứa trong số mẫu kết quả
      chưa được loại bỏ khỏi tập unlabeled samples.
    - CẦN GIẢI QUYẾT VẤN ĐỀ NÀY!!!
"""


# os.environ['KMP_DUPLICATE_LIB_OK']='True'

parser = argparse.ArgumentParser()
parser.add_argument('--query_image_path', type=str, default='', help="store database or search image")
args = parser.parse_args()


# ========================================================
# Color auto-correlogram
# ========================================================
def get_probdist(img, bins, d):
    colours = []
    for c1 in bins:
        for c2 in bins:
            for c3 in bins:
                colours.append([c1, c2, c3])
    
    histogram = np.zeros(64)
    carray = []
    for i in img:
        arr = []
        n = 256 / len(bins)
        for j in i:
            r = (j[0] // n) * n
            g = (j[1] // n) * n
            b = (j[2] // n) * n
            index = colours.index([r, g, b])
            histogram[index] += 1 # number of pixels that have color[index] in whole image
            arr.append(index) # bins value of each pixel in each row
        carray.append(arr) # whole image
    carray = np.array(carray)


    a = np.zeros(shape=(64, len(d), 2)) # Shape = (index of bins, index of distance, track), track: 0: same color, 1: different color
    for i in range(0, len(carray)): # loop for each row => carray[i] : a row in image
        for j in range(len(carray[i])): # loop for each pixel in each row i
            clr = carray[i][j] # bins value of each pixel
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

    for bin_index in range(len(a)): # 64 loops: loop for colors
        histogram_color = histogram[bin_index]
        for j in range(len(a[bin_index])): # 3 loops: loop for distances
            distance = d[j]
            prob = a[bin_index][j][0] / (histogram_color * 8 * distance)
            if np.isnan(prob):
                colour_dist_prob.append(0) 
            else:
                colour_dist_prob.append(prob)
            
    correlogram = np.array(colour_dist_prob)
    return correlogram


# ========================================================
# CNN: VGG19
# ========================================================
class ExtractModel:
    def __init__(self):
        self.model = self.ModelCreator() 

    def ModelCreator(self):
        vgg19_model = VGG19(weights="imagenet")
        extract_model = Model(inputs=vgg19_model.inputs, outputs=vgg19_model.get_layer("fc2").output)
        return extract_model


def preprocessing(img):
    img = Image.fromarray(img)
    img = img.resize((224, 224))
    # img = img.convert('RGB')
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x


def feature_extraction_cnn(image_path, model):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_tensor = preprocessing(image)
    features = model.predict(img_tensor)[0]
    features = features / np.linalg.norm(features)
    return features

# ========================================================
# COLOR HISTOGRAM
# ========================================================

def feature_extraction_color_histogram(image_path):
    image = cv2.imread(image_path) # default flag is BGR
    h, w, c = image.shape
    image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

    """
    Histogram for YCrCb
    n_bins_Y: 8
    n_bins_Cr: 4
    n_bins_Cb: 4
    """
    histogram = cv2.calcHist([image], [0, 1, 2], None, [8, 4, 4], [0, 256, 0, 256, 0, 256]) 
    features = histogram.flatten()
    features /= (h * w)
    # features /= np.linalg.norm(features)
    return features


# ========================================================
# COLOR MOMENTS
# ========================================================
def calculate_mean(a):
    return np.mean(a)

def calculate_std(a):
    mean_a = calculate_mean(a)
    return ((1/len(a)) * (sum((a - mean_a) ** 2))) ** (1/2)

def calculate_skewness(a):
    mean_a = calculate_mean(a)
    s = (1/len(a)) * (sum((a - mean_a) ** 3))
    if s > 0:
        s = s ** (1/3)
    else:
        s = (abs(s) ** (1/3)) * (-1)
    return s

def calculate_moments(image):
    H = image[:, :, 0].flatten()
    S = image[:, :, 1].flatten()
    V = image[:, :, 2].flatten()

    mean_h = calculate_mean(H)
    mean_s = calculate_mean(S)
    mean_v = calculate_mean(V)

    std_h = calculate_std(H)
    std_s = calculate_std(S)
    std_v = calculate_std(V)

    skewness_h = calculate_skewness(H)
    skewness_s = calculate_skewness(S)
    skewness_v = calculate_skewness(V) 

    mean = [mean_h, mean_s, mean_v]
    std = [std_h, std_s, std_v]
    skewness = [skewness_h, skewness_s, skewness_v]

    features = [mean, std, skewness]
    features = np.array(features)
    return features


def split_regions(block):
    h, w = block.shape[:2]
    mid_h = int(h/2)
    mid_w = int(w/2)

    first_region = block[:mid_h,:mid_w]
    second_region = block[:mid_h:,mid_w+1:]
    third_region = block[mid_h+1:,:mid_w]
    forth_region = block[mid_h+1:,mid_w+1:]

    return [first_region, second_region, third_region, forth_region]


def extract_features_one_region(block):
    features_block = calculate_moments(block)
    return features_block

def extract_features_whole_image(image_path):
    """
    Split an image into 4 regions. After that, split each region into 4 sub-regions
    ==> There are: 4x4 (sub-regions) + 4 (regions) + 1 (whole image) to extract features 
    """
    image = cv2.imread(image_path) # default flag is BGR
    image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb) # YCrCb
    image = np.array(image, dtype='float64')

    all_regions = [image]
    image_regions = split_regions(image)
    for region in image_regions:
        all_regions.append(region)
        sub_regions = split_regions(region)
        for sub_region in sub_regions:
            all_regions.append(sub_region)

    features_all_regions = []
    for region in all_regions:
        features_region = extract_features_one_region(region)
        features_all_regions.append(features_region)
    features_all_regions = np.array(features_all_regions)
    features_all_regions = features_all_regions.flatten()
    return features_all_regions


# ========================================================
# SEARCH ENGINE and RELEVANCE FEEDBACK
# ========================================================
def search_image(query_image_path, features_db, paths_db, model):
    features_db = pickle.load(open(features_db, 'rb'))
    paths_db = pickle.load(open(paths_db, 'rb'))

    # cnn
    # query_image_features = feature_extraction_cnn(image_path=query_image_path, model=model)
    # print(query_image_features.shape)

    # color auto-correlogram
    # bins = [0, 64, 128, 192]
    # d = [1, 2, 6]

    # image = cv2.imread(query_image_path)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    # query_image_features = get_probdist(img=image, bins=bins, d=d)

    # color moments
    # query_image_features = extract_features_whole_image(image_path=query_image_path)

    # color histogram
    # query_image_features = feature_extraction_color_histogram(image_path=query_image_path)
    
    # color combine
    # query_image_path = query_image_path.replace('\\','/')
    # path_index = paths_db.index(query_image_path)
    # query_image_features = features_db[path_index]

    # color auto-correlogram + texture GLCM
    query_image_path = query_image_path.replace('\\','/')
    path_index = paths_db.index(query_image_path)
    query_image_features = features_db[path_index]
    
    
    distances = np.linalg.norm(features_db - query_image_features, axis=1)
    K = 100
    indexs = np.argsort(distances)[:K]

    nearest_images = [(features_db[id], paths_db[id], distances[id]) for id in indexs]
    
    return query_image_features, nearest_images


def find_labeled_data(query_image_path, nearest_images):
    class_query_image = query_image_path.split("\\")[-2]

    labeled_data = []

    n_pos = 0
    n_neg = 0
    for img in nearest_images:
        paths_img = img[1]
        x_vector = None
        y_label = None
        if class_query_image in paths_img:
            x_vector = img[0]
            y_label = 1
            n_pos += 1
        else:
            x_vector = img[0]
            y_label = 0
            n_neg += 1
        labeled_data.append((x_vector, y_label, img[1]))
    return labeled_data, n_pos, n_neg


def find_unlabeled_data(nearest_images, features_db, paths_db):
    features_db = pickle.load(open(features_db, 'rb'))
    paths_db = pickle.load(open(paths_db, 'rb'))
    
    paths_nearest_img = []
    for img in nearest_images:
        paths_nearest_img.append(img[1])
    
    unlabeled_img_indexs = []
    for i in range(len(paths_db)):
        path = paths_db[i]
        if path not in paths_nearest_img:
            # unlabeled_img_indexs.append(features_db[i])
            unlabeled_img_indexs.append(i)
    return unlabeled_img_indexs


def compute_DS(svc, unlabeled_data_indexs, features_db):
    features_db = pickle.load(open(features_db, 'rb'))
    DS_arr = []
    for i in range(len(unlabeled_data_indexs)):
        idx = unlabeled_data_indexs[i]
        x = features_db[idx]
        x = x.reshape(1, -1)
        dist = abs(svc.decision_function(x))
        # w_norm = np.linalg.norm(svc.coef_)
        # dist = y / w_norm
        DS_arr.append(dist)
    return DS_arr


def compute_DE(svc, query_image_features, unlabeled_data_indexs, features_db):
    features_db = pickle.load(open(features_db, 'rb'))
    DE_arr = []
    for i in range(len(unlabeled_data_indexs)):
        idx = unlabeled_data_indexs[i]
        x = features_db[idx]
        x = x.reshape(1, -1)
        t = svc.decision_function(x)
        if t >= 0:
            dist = np.linalg.norm(x - query_image_features)
        else:
            dist = int(1e9)
        DE_arr.append(dist)
    return DE_arr


def compute_DSE(unlabeled_data_indexs, n_pos, n_neg, DS_arr, DE_arr):
    DSE_arr = []
    for i in range(len(unlabeled_data_indexs)):
        DS_idx = DS_arr[i]
        DE_idx = DE_arr[i]
        dse = (n_pos/(n_pos+n_neg)) * DS_idx + (1-(n_pos/(n_pos+n_neg))) * DE_idx
        DSE_arr.append(dse)
    return DSE_arr


def svm_active_learning(clf, labeled_data, n_pos, n_neg, unlabeled_data_indexs, query_image_features, query_image_path, nearest_images, features_db, paths_db):

    # labeled_data, n_pos, n_neg = find_labeled_data(query_image_path, nearest_images)
    # unlabeled_data_indexs = find_unlabeled_data(nearest_images, features_db, paths_db)
    temp_unlabeled_data_indexs = unlabeled_data_indexs.copy()
    
    # print(f"n_pos : {n_pos} ====== n_neg : {n_neg}")

    X_train = []
    y_train = []
    for d in labeled_data:
        X_train.append(d[0])
        y_train.append(d[1])

    k = 1000
    # define classifier
    clf.fit(X_train, y_train)

    DS_arr = compute_DS(clf, temp_unlabeled_data_indexs, features_db)
    DE_arr = compute_DE(clf, query_image_features, temp_unlabeled_data_indexs, features_db)

    future_labels = []
    for _ in range(k):
        DSE_arr = compute_DSE(temp_unlabeled_data_indexs, n_pos, n_neg, DS_arr, DE_arr)

        DSE_arr = np.array(DS_arr)
        min_dist_index = np.argmin(DSE_arr) # active learning: find the data point closest from boudary

        idx = temp_unlabeled_data_indexs[min_dist_index]
        future_labels.append(idx) # S* set: data to label
        temp_unlabeled_data_indexs.pop(min_dist_index)
        DS_arr.pop(min_dist_index)
        DE_arr.pop(min_dist_index)
    
    return clf, future_labels


def update_nearest_image(clf, query_image_features, query_image_path, old_nearest_images, future_labels, features_db, paths_db):

    paths_db = pickle.load(open(paths_db, 'rb'))
    features_db = pickle.load(open(features_db, 'rb'))

    class_query_image = query_image_path.split("\\")[-2]

    images = []
    n_pos = 0
    n_neg = 0

    # classify old nearest images: 1 (relevant), 0 (non-relevant)
    for img in old_nearest_images:
        features_img = img[0]
        paths_img = img[1]
        if class_query_image in paths_img:
            n_pos += 1
            images.append((features_img, paths_img, 1, 1)) # images[i]: (features_vector, path_image, rel/non_rel - 1/0, old/new positive - 1/0)
        else:
            n_neg += 1
            images.append((features_img, paths_img, 0, 1))

    # labeling new labels from svm-active-learning algorithm
    for i in future_labels:
        x = features_db[i]
        x = x.reshape(1, -1)
        y_hat = clf.predict(x)[0]
        if y_hat == 1:
            n_pos += 1
            images.append((features_db[i], paths_db[i], 1, 0))
        else:
            n_neg += 1
            images.append((features_db[i], paths_db[i], 0, 0))


    ds_arr = []
    de_arr = []
    dse_arr = []
    old_postive = []


    # save old positive
    for i in range(len(images)):
        img = images[i]
        rel_or_not = img[2] # 1 (rel), 0 (non-rel)
        old_or_not = img[3] # 1 (old), 0 (new)
        if rel_or_not == 1 and old_or_not == 1: # old positive
            old_postive.append(i)

    # compute DS
    for img in images:
        features = img[0]
        features = features.reshape(1, -1)
        path = img[1]
        dist = abs(clf.decision_function(features))
        # w_norm = np.linalg.norm(svc.coef_)
        # dist = y / w_norm
        ds_arr.append(dist)

    # compute DE
    for img in images:
        features = img[0]
        features = features.reshape(1, -1)
        # t = clf.decision_function(features)
        if img[2] == 1:
            dist = np.linalg.norm(features - query_image_features)
        else:
            dist = int(1e9)
        de_arr.append(dist)

    # compute DSE
    for i in range(len(images)):
        if i in old_postive:
            alpha = 1/4
        else:
            alpha = 4
        DS_idx = ds_arr[i]
        DE_idx = de_arr[i]
        dse = 0.3 * DS_idx + 0.7 * DE_idx
        # dse = (n_pos/(n_pos+n_neg)) * DS_idx + (1-(n_pos/(n_pos+n_neg))) * DE_idx
        dse = dse * alpha # ensure that old positive will be presented at first
        dse_arr.append(dse)

    dse_arr = np.array(dse_arr)
    dse_arr = dse_arr.reshape(-1)
    K = 100
    indexs = np.argsort(dse_arr)[:K]

    nearest_images = [(images[id][0], images[id][1], dse_arr[id]) for id in indexs]
    return nearest_images


def update_current_labeled_data(query_image_path, current_labeled_data, current_n_pos, current_n_neg, new_nearest_images):
    temp_labeled_data_set, temp_n_pos, temp_n_neg = find_labeled_data(query_image_path, new_nearest_images)

    arr_path_current_labeled_data = []
    for labeled_data in current_labeled_data:
        path = labeled_data[2]
        arr_path_current_labeled_data.append(path)

    counter = 0
    for temp_labeled_data in temp_labeled_data_set:
        pos_or_neg = temp_labeled_data[1]
        path_temp_labeled_data = temp_labeled_data[2]
        if path_temp_labeled_data not in arr_path_current_labeled_data:
            current_labeled_data.append(temp_labeled_data)
            if pos_or_neg == 1:
                current_n_pos += 1
            elif pos_or_neg == 0:
                current_n_neg += 1
            counter += 1
    # print(counter)
    return current_labeled_data, current_n_pos, current_n_neg


def update_current_unlabeled_data_indices(paths_db, current_labeled_data, current_unlabeled_data_indices):
    # print(f"==== {len(current_unlabeled_data_indices)}")
    paths_db = pickle.load(open(paths_db, 'rb'))

    for labeled_data in current_labeled_data:
        path = labeled_data[2]
        paths_db_index = paths_db.index(path)
        if paths_db_index in current_unlabeled_data_indices:
            current_unlabeled_data_indices.remove(paths_db_index)
    return current_unlabeled_data_indices


def plot_result(nearest_images):
    """
    PLOT
    """
    # grid_size = int(math.sqrt(K))
    grid_row = 5
    grid_col = 20
    fig, axes = plt.subplots(grid_row, grid_col, figsize=(15, 12))
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
    # vgg19_model = ExtractModel().model
    # vgg19_model.summary()

    DATA_FOLDER = 'P:/cbir/DATA/corel/CorelDB/'
    features_db_file = "./database/features_autocorrelogram_glcm.pkl"
    file_path_db_file = "./database/paths.pkl"

    # data_non_rf = {}
    data_2 = {}
    data_3 = {}
    # data_5 = {}
    # total_average_precision_non_rf = 0
    total_average_precision_2 = 0
    total_average_precision_3 = 0
    # total_average_precision_5 = 0
    for sub_folder in os.listdir(DATA_FOLDER):
        print()
        print("==============> ", sub_folder)
        sub_folder_path = DATA_FOLDER + sub_folder + '/'
        chk = 0

        # total_precision_non_rf = 0
        total_precision_2 = 0
        total_precision_3 = 0
        # total_precision_5 = 0

        for file_name in os.listdir(sub_folder_path):
            if chk == 2:
                break
            if file_name.endswith('.jpg'):
                kernel = 'rbf'
                clf = svm.SVC(kernel=kernel)
                print(f"Class {sub_folder} - Image {chk+1}")
                chk += 1
                query_image_path = sub_folder_path + file_name
                query_image_path = query_image_path.replace('/','\\')

                # Search image
                query_image_features, nearest_images = search_image(query_image_path=query_image_path,features_db=features_db_file, paths_db=file_path_db_file, model=None)
                # plot_result(nearest_images)

                labeled_data_set, n_pos, n_neg = find_labeled_data(query_image_path, nearest_images)
                unlabeled_data_set_indices = find_unlabeled_data(nearest_images, features_db=features_db_file, paths_db=file_path_db_file)
                # print(f"n_pos : {n_pos}%")
                # total_precision_non_rf += n_pos

                # """
                for i in range(3):
                    print(f"====> RF {i+1}:")
                    clf, future_labels = svm_active_learning(clf, labeled_data_set, n_pos, n_neg, unlabeled_data_set_indices, query_image_features, query_image_path, nearest_images, features_db_file, file_path_db_file)
                    nearest_images = update_nearest_image(clf, query_image_features, query_image_path, nearest_images, future_labels, features_db_file, file_path_db_file)
                    labeled_data_set, n_pos, n_neg = update_current_labeled_data(query_image_path, labeled_data_set, n_pos, n_neg, nearest_images)
                    unlabeled_data_set_indices = update_current_unlabeled_data_indices(file_path_db_file, labeled_data_set, unlabeled_data_set_indices)
                    if i == 1:
                        total_precision_2 += n_pos
                    elif i == 2:
                        total_precision_3 += n_pos
                    # print(f"n_pos : {n_pos}%") 
                # total_precision_5 += n_pos
                # """

        # Non-RF
    #     average_precision_non_rf = total_precision_non_rf / chk
    #     data_non_rf[f'{sub_folder}'] = average_precision_non_rf
    #     total_average_precision_non_rf += average_precision_non_rf
    # mean_average_precision_non_rf = total_average_precision_non_rf / 80
    # print("========= MAP = ", mean_average_precision_non_rf)
    # data_non_rf = dict(sorted(data_non_rf.items(), key=lambda item: item[1], reverse=True))
    # df_non_rf = pd.DataFrame(data_non_rf, index=[0])
    # df_non_rf.to_csv('history_non_rf_autocorrelogram_glcm.csv')

        # RF
    # """        
        average_precision_2 = total_precision_2 / chk
        average_precision_3 = total_precision_3 / chk
        # average_precision_5 = total_precision_5 / chk
        data_2[f'{sub_folder}'] = average_precision_2
        data_3[f'{sub_folder}'] = average_precision_3
        # data_5[f'{sub_folder}'] = average_precision_5
        total_average_precision_2 += average_precision_2
        total_average_precision_3 += average_precision_3
        # total_average_precision_5 += average_precision_5
    mean_average_precision_2 = total_average_precision_2 / 80
    mean_average_precision_3 = total_average_precision_3 / 80
    # mean_average_precision_5 = total_average_precision_5 / 80
    print("MAP RF 2: ", mean_average_precision_2)
    print("MAP RF 3: ", mean_average_precision_3)
    # print("MAP RF 5: ", mean_average_precision_5)

    data_2 = dict(sorted(data_2.items(), key=lambda item: item[1], reverse=True))
    data_3 = dict(sorted(data_3.items(), key=lambda item: item[1], reverse=True))
    # data_5 = dict(sorted(data_5.items(), key=lambda item: item[1], reverse=True))
    df_2 = pd.DataFrame(data_2, index=[0])
    df_3 = pd.DataFrame(data_3, index=[0])
    # df_5 = pd.DataFrame(data_5, index=[0])
    df_2.to_csv('history_rf_2_color_autocorrelogram_glcm.csv')
    df_3.to_csv('history_rf_3_color_autocorrelogram_glcm.csv')
    # df_5.to_csv('history_rf_5.csv')
    # """
                





"""
    # query_image_path = args.query_image_path

    # Search image

    query_image_path = args.query_image_path

    features_db_file = "./database/features_color_histogram.pkl"
    file_path_db_file = "./database/paths.pkl"
    query_image_features, nearest_images = search_image(query_image_path=query_image_path,features_db=features_db_file, paths_db=file_path_db_file, model=None)
    # plot_result(nearest_images)

    labeled_data_set, n_pos, n_neg = find_labeled_data(query_image_path, nearest_images)
    unlabeled_data_set_indices = find_unlabeled_data(nearest_images, features_db=features_db_file, paths_db=file_path_db_file)
    plot_result(nearest_images)
    print(f"n_pos : {n_pos} ====== n_neg : {n_neg}")
    # print(len(labeled_data_set))
    # print(len(unlabeled_data_set_indices))

    for i in range(2):
        print(f"====> RF {i+1}:")
        clf, future_labels = svm_active_learning(clf, labeled_data_set, n_pos, n_neg, unlabeled_data_set_indices, query_image_features, query_image_path, nearest_images, features_db_file, file_path_db_file)
        nearest_images = update_nearest_image(clf, query_image_features, query_image_path, nearest_images, future_labels, features_db_file, file_path_db_file)
        labeled_data_set, n_pos, n_neg = update_current_labeled_data(query_image_path, labeled_data_set, n_pos, n_neg, nearest_images)
        unlabeled_data_set_indices = update_current_unlabeled_data_indices(file_path_db_file, labeled_data_set, unlabeled_data_set_indices)
        print(f"n_pos : {n_pos} ====== n_neg : {n_neg}")
        # print(len(labeled_data_set))
        # print(len(unlabeled_data_set_indices))
        # plot_result(nearest_images)
    
    # print(f"n_pos : {n_pos} ====== n_neg : {n_neg}")
    plot_result(nearest_images)
"""