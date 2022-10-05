import os
from flask import Flask, render_template
from flask_mail import Mail
from app.model.UserDB import UserDB


def page_not_found(e):
    return render_template('exceptionview/404.html'), 404


def list_categories():
    objDb = UserDB()
    sql = " SELECT DISTINCT Image_Class " \
            " FROM db_cbir.tb_image " \
            " ORDER BY Image_Class ASC; "

    val = []
    res = objDb.query_get(sql, val)
    image_categories = []
    for _ in res:
        for x in _:
            image_categories.append(x)
    return dict(image_categories=image_categories)


app = Flask(__name__)
app.secret_key = "f23sl398jhfei"

import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

UPLOAD_FOLDER = './app/static/img_Uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.mkdir(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

app.config['DATA_FOLDER'] = "./app/static/CorelDatabase/"


app.config['MAIL_SERVER']='smtp.gmail.com'
app.config['MAIL_PORT'] = 465
app.config['MAIL_USERNAME'] = 'cbir.project30@gmail.com'
app.config['MAIL_PASSWORD'] = 'kmqwdrjjvpvybcxs'
app.config['MAIL_DEFAULT_SENDER'] = 'CBIR'
app.config['MAIL_USE_TLS'] = False
app.config['MAIL_USE_SSL'] = True
mail = Mail(app)

app.register_error_handler(404, page_not_found) # if url not exist
app.context_processor(list_categories) # declare a variable for all templates

from app.controller import HomeController
from app.controller import LoginController
from app.controller import AdminController
from app.controller import UserController