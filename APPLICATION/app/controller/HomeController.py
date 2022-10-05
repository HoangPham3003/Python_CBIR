from unicodedata import category
from app import app

import os
import cv2 as cv
import numpy as np
from werkzeug.utils import secure_filename
from flask import request, render_template, url_for, jsonify, redirect, session

from ..model.UserDB import UserDB
from .SystemController import SystemController


ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'tif'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def rgb_to_gray(input_path, file_name):
    # Processing img uploaded
    img = cv.imread(input_path)         # Read image
    if img is None:
        print("Không tìm thấy file ảnh")

    # Processing img
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Create link img render
    fname = file_name.split(".")[0]
    output_path = "./app/static/img_Render/" + fname + "_RgbToGray.jpg"
    cv.imwrite(output_path, img)
    send_path = "../static/img_Render/" + fname + "_RgbToGray.jpg"

    cv.destroyAllWindows()
    return send_path


def get_images_by_class(image_class):
    objDb = UserDB()
    sql = "SELECT * FROM db_cbir.tb_image WHERE Image_Class = %s;"
    val = [image_class]
    res = objDb.query_get(sql, val)
    return res


def get_images_path_per_page(start, end):
    objDb = UserDB()
    sql = "SELECT * FROM db_cbir.tb_image LIMIT %s, %s;"
    val = (start, end)
    res = objDb.query_get(sql, val)
    return res


def count_total_images():
    objDb = UserDB()
    sql = "SELECT COUNT(Image_Id) FROM db_cbir.tb_image;"
    val = ()
    res = objDb.query_get(sql, val)
    return res


@app.route('/gray', methods=['POST'])
def gray():
    file = request.files['file']
    if file and allowed_file(file.filename):
        fileName = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], fileName))

        file_path = "./app/static/img_Uploads/" + str(fileName)
        res = rgb_to_gray(file_path, fileName)
        res = str(res)
        return jsonify(data=res)


@app.route("/images/category/<string:image_class>", endpoint="show_image_category")
@SystemController.check_acl
def show_image_category(image_class):
    # if logged in before
    if 'auth' in session:
        if session['auth']['Role_Name'] == 'admin':
            return redirect(f"/admin/images/category/{image_class}")
        elif session['auth']['Role_Name'] == 'user':
            return redirect(f"/user/images/category/{image_class}")
    images = get_images_by_class(image_class)
    db_images = []
    for info in images:
        image_class = info[1]
        image_name = info[2]
        image_info = {
            "class": image_class,
            "name": image_name
        }
        db_images.append(image_info)
    return render_template('unlogedview/unloged_images_category.html', db_images=db_images, category=image_class)


@app.route("/page/<int:page_index>", endpoint="show_image_page")
@SystemController.check_acl
def show_image_page(page_index):
    # if logged in before
    if 'auth' in session:
        if session['auth']['Role_Name'] == 'admin':
            return redirect(f"/admin/home/page/{page_index}")
        elif session['auth']['Role_Name'] == 'user':
            return redirect(f"/user/home/page/{page_index}")
    if page_index == 1:
        return redirect("/")
    no_images_per_page = 200
    total_no_images = count_total_images()[0][0]
    total_no_pages = int(np.ceil(total_no_images / no_images_per_page))
    current_page = page_index

    if current_page > total_no_pages:
        return redirect(f"/page/{total_no_pages}")

    start = (current_page - 1) * no_images_per_page
    images = get_images_path_per_page(start, no_images_per_page)
    db_images = []
    for info in images:
        image_class = info[1]
        image_name = info[2]
        image_info = {
            "class": image_class,
            "name": image_name
        }
        db_images.append(image_info)
    return render_template('unlogedview/unloged_home.html', db_images=db_images)


@app.route("/profile", endpoint="profile")
def profile():
    if 'auth' in session:
        if session['auth']['Role_Name'] == 'admin':
            return redirect("/admin/profile")
        elif session['auth']['Role_Name'] == 'user':
            return redirect("/user/profile")
    return render_template('exceptionview/404.html')


@app.route("/about-us", endpoint="aboutus")
@SystemController.check_acl
def aboutus():
    if 'auth' in session:
        if session['auth']['Role_Name'] == 'admin':
            return redirect("/admin/about-us")
        elif session['auth']['Role_Name'] == 'user':
            return redirect("/user/about-us")
    return render_template('unlogedview/unloged_aboutus.html')


@app.route("/favicon.ico")
def favicon():
    return "", 200


@app.route("/")
@SystemController.check_acl
def home():
    # if logged in before
    if 'auth' in session:
        if session['auth']['Role_Name'] == 'admin':
            return redirect("/admin/home")
        elif session['auth']['Role_Name'] == 'user':
            return redirect("/user/home")
    images = get_images_path_per_page(0, 200)
    db_images = []
    for info in images:
        image_class = info[1]
        image_name = info[2]
        image_info = {
            "class": image_class,
            "name": image_name
        }
        db_images.append(image_info)
    return render_template('unlogedview/unloged_home.html', db_images=db_images)

