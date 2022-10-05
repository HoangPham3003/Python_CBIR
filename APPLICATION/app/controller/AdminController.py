from app import app

import os
import re
import cv2 as cv
import numpy as np
from werkzeug.utils import secure_filename
from flask import request, render_template, url_for, jsonify, redirect, session

from ..model.UserDB import UserDB
from .SystemController import SystemController
from .HomeController import get_images_path_per_page, count_total_images


@app.route("/admin/home/page/<int:page_index>", endpoint='admin_home_page')
@SystemController.login_required
@SystemController.check_acl
def show_image_page_admin(page_index):
    if page_index == 1:
        return redirect("/")
    no_images_per_page = 200
    total_no_images = count_total_images()[0][0]
    total_no_pages = int(np.ceil(total_no_images / no_images_per_page))
    current_page = page_index

    if current_page > total_no_pages:
        return redirect(f"admin/home/page/{total_no_pages}")

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
    return render_template('logedview/ACLview/adminview/admin_home.html', db_images=db_images)


@app.route("/admin/home", endpoint='admin_home', methods=['GET'])
@SystemController.login_required
@SystemController.check_acl
def admin_home():
    # if logged in before
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
    return render_template('logedview/ACLview/adminview/admin_home.html', db_images=db_images)
    
