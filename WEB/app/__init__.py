import os
import numpy as np

from flask import Flask, render_template
from app.model.UserDB import UserDB


# def page_not_found(e):
#     return render_template('exceptionview/404.html'), 404


# def list_categories():
#     objDb = UserDB()
#     sql = " SELECT DISTINCT Image_Class " \
#             " FROM db_cbir.tb_image " \
#             " ORDER BY Image_Class ASC; "

#     val = []
#     res = objDb.query_get(sql, val)
#     image_categories = []
#     for _ in res:
#         for x in _:
#             image_categories.append(x)
#     return dict(image_categories=image_categories)


def get_features_db():
    objDb = UserDB()
    sql = "SELECT * FROM db_cbir_v2.tb_features"
    val = []
    res = objDb.query_get(sql, val)
    features_db = []
    for r in res:
        str_features = r[1]
        features = [float(x) for x in str_features.split(',')]
        features = np.array(features)
        features_db.append(features)
    return features_db

def get_paths_db():
    objDb = UserDB()
    sql = "SELECT * FROM db_cbir_v2.tb_image"
    val = []
    res = objDb.query_get(sql, val)
    paths_db = []
    for r in res:
        image_class = r[1]
        image_name = r[2]
        image_path = "./app/static/CorelDatabase/" + image_class + '/' + image_name
        paths_db.append(image_path)
    return paths_db


app = Flask(__name__)
app.secret_key = "f23sl398jhfei"

# import logging
# log = logging.getLogger('werkzeug')
# log.setLevel(logging.ERROR)

UPLOAD_FOLDER = './app/static/img_Uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.mkdir(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

app.config['DATA_FOLDER'] = "./app/static/CorelDatabase/"




features_db = get_features_db()
paths_db = get_paths_db()
app.config['FEATURES_DB'] = features_db
app.config['PATHS_DB'] = paths_db

# app.register_error_handler(404, page_not_found) # if url not exist
# app.context_processor(get_features_db) # declare a variable for all templates

from app.controller import HomeController


