import os
import random
import mysql.connector
from tqdm import tqdm

from .ConnectDB import DB

import pickle


class UserDB:
    def __init__(self):
        self.db = DB().connect_db()

    def query_set(self, sql, val):
        try:
            cursor = self.db.cursor()
            cursor.execute(sql, val)
            self.db.commit()
            cursor.close()
            return True
        except mysql.connector.Error:
            return False

    def query_get(self, sql, val):
        try:
            cursor = self.db.cursor()
            cursor.execute(sql, val)
            res = cursor.fetchall()
            cursor.close()
            return res
        except mysql.connector.Error:
            return False


def create_tb_image():
    objDB = UserDB()
    db = []
    data_folder = "P:/cbir/DATA/corel/CorelDB/"
    for sub_folder in tqdm(os.listdir(data_folder)):
        sub_folder_path = data_folder + sub_folder + '/'
        for file_name in os.listdir(sub_folder_path):
            if file_name.endswith('.jpg'):
                image_info = {
                    "class": sub_folder,
                    "name": file_name
                }
                db.append(image_info)

    # random.shuffle(db)
    for image_info in tqdm(db):
        sql = "INSERT INTO DB_CBIR_V2.TB_Image (Image_Class, Image_Name) " \
            "VALUES (%s, %s); "
        val = (image_info['class'], image_info['name'])
        res = objDB.query_set(sql, val)


def create_tb_features():
    objDB = UserDB()

    # Insert value
    features_db_file = "features.pkl"
    features_db = pickle.load(open(features_db_file, 'rb'))

    for i in tqdm(range(len(features_db))[1:]):
        s = ','
        str_features = s.join([str(x) for x in features_db[i]])
        sql = "INSERT INTO DB_CBIR_V2.TB_Features " \
            "VALUES (%s, %s); "
        val = (i+1, str_features)
        res = objDB.query_set(sql, val)
    
    print("DONE")



def get_paths_db():
    objDb = UserDB()
    sql = "SELECT * FROM db_cbir_v2.tb_image"
    val = []
    res = objDb.query_get(sql, val)
    paths_db = []
    for r in res[:1]:
        image_class = r[1]
        image_name = r[2]
        image_path = "./app/static/CorelDatabase/" + image_class + '/' + image_name
        paths_db.append(image_path)
    return paths_db

if __name__ == '__main__':
    # get_paths_db()
    # get_features_db()
    pass
    # create_tb_image()
    # create_tb_features()
    # features_db = select_features_db()
    # print(features_db[0])
