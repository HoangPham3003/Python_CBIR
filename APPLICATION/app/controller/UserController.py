from app import app

import os
import re
import json
import datetime
import cv2 as cv
import numpy as np
from werkzeug.utils import secure_filename
from flask import request, render_template, url_for, jsonify, redirect, session

from ..model.UserDB import UserDB
from .SystemController import SystemController
from .HomeController import get_images_path_per_page, count_total_images, get_images_by_class



@app.route("/user/home/page/<int:page_index>", endpoint='user_home_page')
@SystemController.login_required
@SystemController.check_acl
def show_image_page_user(page_index):
    if page_index == 1:
        return redirect("/")
    no_images_per_page = 200
    total_no_images = count_total_images()[0][0]
    total_no_pages = int(np.ceil(total_no_images / no_images_per_page))
    current_page = page_index

    if current_page > total_no_pages:
        return redirect(f"user/home/page/{total_no_pages}")

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
    return render_template('logedview/ACLview/userview/user_home.html', db_images=db_images)


@app.route("/user/home", endpoint='user_home', methods=['GET'])
@SystemController.login_required
@SystemController.check_acl
def user_home():
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
    return render_template('logedview/ACLview/userview/user_home.html', db_images=db_images)


@app.route("/user/profile", endpoint='user_profile', methods=['GET', 'POST'])
@SystemController.login_required
@SystemController.check_acl
def user_profile():
    objDb = UserDB()
    user_email = session['auth']['User_Email']

    sql = " SELECT User_FirstName, User_LastName, User_Dob, User_Gender, User_Email, User_Phone" \
            " FROM db_cbir.tb_user WHERE User_Email = %s; "
    val = [user_email]
    user = objDb.query_get(sql, val)

    user_info = {
        "User_FirstName": user[0][0],
        "User_LastName": user[0][1],
        "User_Dob": str(user[0][2]),
        "User_Gender": user[0][3],
        "User_Email": user[0][4],
        "User_Phone": user[0][5]
    }
    

    check = True
    error = []
    message = []
    if request.method == 'POST' and 'input_update_profile' in request.form:
        input_data = json.loads(request.form['input_update_profile'])

        first_name = input_data['fname'].strip()
        last_name = input_data['lname'].strip()
        dob = input_data['dob'].strip()
        gender = input_data['gender']
        phone = input_data['phone'].strip()

        if first_name == '' or last_name == '' or dob == '' or gender == '' or phone == '':
            check = False
            err_text = "Please enter enough information!"
            error.append(err_text)

        if check:
            # check email existing
            sql = "UPDATE db_cbir.tb_user " \
                  "SET User_FirstName = %s, User_LastName = %s, User_Dob = %s, User_Gender = %s, User_Phone = %s " \
                  "WHERE User_Email = %s;"
            val = [first_name, last_name, dob, gender, phone, user_info['User_Email']]
            res = objDb.query_set(sql, val)
            
            if res:
                msg_text = "Update profile successully!"
                message.append(msg_text)
                return jsonify(message=message)
        else:
            return jsonify(error=error)

    elif request.method == 'POST' and 'input_change_pwd' in request.form:
        input_data = json.loads(request.form['input_change_pwd'])

        current_pwd = input_data['current_pwd'].strip()
        new_pwd = input_data['new_pwd'].strip()
        retyped_new_pwd = input_data['retyped_new_pwd'].strip()

        sql = "SELECT User_PassWord " \
                "FROM db_cbir.tb_user " \
                "WHERE User_Email = %s;"
        val = [user_info['User_Email']]
        auth_current_pwd = objDb.query_get(sql, val)[0][0]

        check_password = r"^\S{5,20}$"

        if current_pwd == '' or new_pwd == '' or retyped_new_pwd == '':
            check = False
            err_text = "Please enter enough information!"
            error.append(err_text)
        elif current_pwd != auth_current_pwd:
            check = False
            err_text = "Wrong current password!"
            error.append(err_text)
        elif re.search(check_password, new_pwd) is None:
            check = False
            err_text = "Password is from 5 to 20 characters and do not include space blank!"
            error.append(err_text)
        elif new_pwd == auth_current_pwd:
            check = False
            err_text = "New password need to be different from current password!"
            error.append(err_text)
        elif new_pwd != retyped_new_pwd:
            check = False
            err_text = "Retyped new password need to be same as new password!"
            error.append(err_text)
        
        
        if check:
            # check email existing
            sql = "UPDATE db_cbir.tb_user " \
                  "SET User_PassWord = %s " \
                  "WHERE User_Email = %s;"
            val = [retyped_new_pwd, user_info['User_Email']]
            res = objDb.query_set(sql, val)
            
            if res:
                msg_text = "Change password successully! You will be signed out after 1s!"
                message.append(msg_text)
                return jsonify(message=message)
        else:
            return jsonify(error=error)
    
    return render_template('logedview/ACLview/userview/user_profile.html', user_info=user_info)


@app.route("/user/images/category/<string:image_class>", endpoint="show_image_category_user")
@SystemController.login_required
@SystemController.check_acl
def show_image_category_user(image_class):
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
    return render_template('logedview/loged_images_category.html', db_images=db_images, category=image_class)


@app.route("/user/about-us", endpoint="aboutus_user")
@SystemController.login_required
@SystemController.check_acl
def aboutus_user():
    return render_template('logedview/loged_aboutus.html')