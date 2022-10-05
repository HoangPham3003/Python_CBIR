import os
import random
import mysql.connector
from tqdm import tqdm

from .ConnectDB import DB


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

    random.shuffle(db)
    for image_info in tqdm(db):
        sql = "INSERT INTO DB_CBIR.TB_Image (Image_Class, Image_Name) " \
            "VALUES (%s, %s); "
        val = (image_info['class'], image_info['name'])
        res = objDB.query_set(sql, val)


if __name__ == '__main__':
    create_tb_image()
    
