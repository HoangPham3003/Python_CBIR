from app import app, mail

import os
import re
import json
import time
from datetime import timedelta
import string
import random
import cv2 as cv
import numpy as np
from werkzeug.utils import secure_filename
from flask import request, render_template, url_for, jsonify, redirect, session
from flask_mail import Message

from ..model.UserDB import UserDB
from .SystemController import SystemController


@app.route("/forgot-password", endpoint='forgot_password', methods=['GET', 'POST'])
@SystemController.check_acl
def forgot_password():
    check = True
    error = []
    message = []

    # if logged in before
    if 'auth' in session:
        if session['auth']['Role_Name'] == 'admin':
            return redirect("/admin/home")
        elif session['auth']['Role_Name'] == 'user':
            return redirect("/user/home")
            
    if request.method == 'POST' and 'input_forgot_password' in request.form:
        input_data = json.loads(request.form['input_forgot_password'])

        email = input_data['email'].strip()

        # check validation of input
        check_email = r"^[A-Za-z0-9_.]{1,50}@gmail.com$"

        if re.search(check_email, email) is None:
            check = False
            err_text = "Email has form abc@gmail.com!"
            error.append(err_text)

        if check:
            # Create a UserDB object
            objDb = UserDB()

            # check email existing
            sql = "SELECT * " \
                  "FROM db_cbir.tb_user " \
                  "WHERE tb_user.User_Email = %s;"
            val = [email]
            user = objDb.query_get(sql, val)

            if not len(user) > 0:
                err_text = "Email doesn't exist!"
                error.append(err_text)
                return jsonify(error=error)
            else:
                
                characters = list(string.ascii_letters) + list(string.digits) + list(string.punctuation) 

                new_password = ""
                len_new_password = random.randint(10, 12)
                for i in range(len_new_password):
                    index_char = random.randint(0, len(characters)-1)
                    c = characters[index_char]
                    new_password += c
                
                sql = " UPDATE db_cbir.tb_user " \
                        "SET User_PassWord = %s " \
                        "WHERE User_Email = %s;"
                val = [new_password, email]

                res = objDb.query_set(sql, val)
                if res:
                    body = f"Your password was reset! Your new password: {new_password}"
                    msg = Message(subject='[CBIR - EMAIL VERIFICATION]', body=body, recipients=[email])
                    mail.send(msg)
                    
                    msg_text = "A new password was sent to your email! <a href='/signin'>Sign in now</a>"
                    message.append(msg_text)
                    return jsonify(message=message)
                else:
                    err_text = "Error in creating new password!"
                    error.append(err_text)
                    return jsonify(error=error) 
        else:
            return jsonify(error=error)
    return render_template('unlogedview/loginview/forgot_password.html')


@app.route("/signin", endpoint='signin', methods=['GET', 'POST'])
@SystemController.check_acl
def signin():
    check = True
    error = []
    # if logged in before
    if 'auth' in session:
        if session['auth']['Role_Name'] == 'admin':
            return redirect("/admin/home")
        elif session['auth']['Role_Name'] == 'user':
            return redirect("/user/home")

    # if have not logged in yet
    if request.method == 'POST' and request.form['submit'] is not None:
        email = request.form['email'].strip()
        password = request.form['password'].strip()

        # check validation of input
        check_email = r"^[A-Za-z0-9_.]{1,50}@gmail.com$"
        check_password = r"^\S{5,20}$"

        if re.search(check_email, email) is None:
            check = False
            err_text = "Email has form abc@gmail.com !!!"
            error.append(err_text)
        if re.search(check_password, password) is None:
            check = False
            err_text = "Password is from 5 to 20 characters and do not include space blank!!!"
            error.append(err_text)

        if check:
            # Create a UserDB object
            obj_db = UserDB()

            # check email existing
            sql = "SELECT User_Id, User_FirstName, User_LastName, User_Dob, User_Gender, User_Email, User_Phone, User_PassWord, Role_Name " \
                  "FROM db_cbir.tb_user " \
                  "LEFT JOIN db_cbir.tb_role " \
                  "ON tb_user.Role_Id_fk = tb_role.Role_Id " \
                  "WHERE tb_user.User_Email = %s;"
            val = [email]
            user = obj_db.query_get(sql, val)

            if not user:
                error_text = "Email is not existing !!!"
                error.append(error_text)
                return render_template('unlogedview/loginview/login.html', error=error)
            else:
                # check correct password
                info_user = {
                    # "User_Id": user[0][0],
                    # "User_FirstName": user[0][1],
                    # "User_LastName": user[0][2],
                    # "User_Dob": str(user[0][3]),
                    # "User_Gender": user[0][4],
                    "User_Email": user[0][5],
                    # "User_Phone": user[0][6],
                    "User_PassWord": user[0][7],
                    "Role_Name": user[0][8]
                }
                
                if password == info_user['User_PassWord']:
                    info_user.pop("User_PassWord")
                    # Get pms name by role name
                    sql = "SELECT Pms_Name FROM db_cbir.tb_pms " \
                          "LEFT JOIN db_cbir.tb_role_pms " \
                          "ON tb_pms.Pms_Id = tb_role_pms.Pms_Id_fk " \
                          "LEFT JOIN db_cbir.tb_role " \
                          "ON tb_role_pms.Role_Id_fk = tb_role.Role_Id " \
                          "WHERE Role_Name = %s;"
                    val = [info_user['Role_Name']]
                    pms_names = obj_db.query_get(sql, val)
                    info_user['Pms_Name'] = []
                    for pms_tuple in pms_names:
                        for pms in pms_tuple:
                            info_user['Pms_Name'].append(pms)

                    if "check_remember" in request.form:
                        session.permanent = True
                        app.permanent_session_lifetime = timedelta(days=30)
                    session['auth'] = info_user
                    
                    if info_user['Role_Name'] == 'admin':
                        return redirect("/admin/home")
                    elif info_user['Role_Name'] == 'user':
                        return redirect('/user/home')
                else:
                    error_text = "Wrong password !!!"
                    error.append(error_text)
                    return render_template('unlogedview/loginview/login.html', error=error)
        else:
            return render_template('unlogedview/loginview/login.html', error=error)
    return render_template('unlogedview/loginview/login.html', error=None)


@app.route('/signout', endpoint='signout')
@SystemController.check_acl
def logout():
    session.clear()
    return redirect('/signin')


@app.route("/register",endpoint='register', methods=['GET', 'POST'])
@SystemController.check_acl
def register():
    obj_db = UserDB()

    check = True
    error = []
    message = []
    # if logged in before
    if 'auth' in session:
        if session['auth']['Role_Name'] == 'admin':
            return redirect("/admin/home")
        elif session['auth']['Role_Name'] == 'user':
            return redirect("/user/home")

    # if have not logged in yet
    if request.method == 'POST' and 'input_register' in request.form:
        input_data = json.loads(request.form['input_register'])

        first_name = input_data['fname'].strip()
        last_name = input_data['lname'].strip()
        dob = input_data['dob'].strip()
        gender = input_data['gender']
        em = input_data['email'].strip()
        phone = input_data['phone'].strip()
        pwd = input_data['pwd'].strip()
        retyped_pwd = input_data['retyped_pwd'].strip()

        # check validation of input
        check_email = r"^[A-Za-z0-9_.]{1,50}@gmail.com$"
        check_password = r"^\S{5,20}$"

        if first_name == '' or last_name == '' or dob == '' or gender == '' \
            or em == '' or phone == '' or pwd == '' or retyped_pwd == '':
            check = False
            err_text = "Please enter enough information!"
            error.append(err_text)
        elif re.search(check_email, em) is None:
            check = False
            err_text = "Email has form abc@gmail.com!"
            error.append(err_text)
        elif re.search(check_password, pwd) is None:
            check = False
            err_text = "Password is from 5 to 20 characters and do not include space blank!"
            error.append(err_text)
        elif pwd != retyped_pwd:
            check = False
            err_text = "Retyped password need to be same as password!"
            error.append(err_text)

        if check:
            # Create a UserDB object

            # check email existing
            sql = "SELECT * " \
                  "FROM db_cbir.tb_user " \
                  "WHERE tb_user.User_Email = %s;"
            val = [em]
            user = obj_db.query_get(sql, val)
            if len(user) > 0:
                err_text = "Email existed!"
                error.append(err_text)
                return jsonify(error=error)
            else:
                info_register = {
                    "User_FirstName": first_name,
                    "User_LastName": last_name,
                    "User_Dob": dob,
                    "User_Gender": gender,
                    "User_Email": em,
                    "User_Phone": phone,
                    "User_PassWord": retyped_pwd
                }
                session['info_register'] = info_register

                verification_code = random.randint(100000, 999999)
                session['AUTH_VERIFICATION_CODE'] = verification_code # save code for later
                
                body = f"Your verification code: {verification_code}"
                msg = Message(subject='[CBIR - EMAIL VERIFICATION]', body=body, recipients=[em])
                mail.send(msg)

                now = time.time()
                session['start_time'] = now
                session['end_time'] = now + 180
                
                msg_text = "Accepted input data"
                message.append(msg_text)
                return jsonify(message=message)
        else:
            return jsonify(error=error)
    elif request.method == 'POST' and 'email_verification' in request.form:
        input_data = json.loads(request.form['email_verification'])
        input_verification_code = input_data['input_code'].strip()
        now = time.time()
        if now > session['end_time']: # if code is expired
            # resend code
            verification_code = random.randint(100000, 999999)
            session['AUTH_VERIFICATION_CODE'] = verification_code # save code for later
            
            body = f"Your verification code: {verification_code}"
            msg = Message(subject='[CBIR - EMAIL VERIFICATION]', body=body, recipients=[session['info_register']['User_Email']])
            mail.send(msg)

            now = time.time()
            session['start_time'] = now
            session['end_time'] = now + 20

            err_text = "Verification code is expired!"
            error.append(err_text)
            return jsonify(error=error)
        elif int(session['AUTH_VERIFICATION_CODE']) != int(input_verification_code):
            err_text = "Wrong verification code!"
            error.append(err_text)
            return jsonify(error=error)
        elif int(session['AUTH_VERIFICATION_CODE']) == int(input_verification_code):
            # insert new user
            sql = " INSERT INTO db_cbir.tb_user (User_FirstName, User_LastName, User_Dob, User_Gender, User_Email, User_Phone, User_PassWord, Role_Id_fk) " \
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"

            first_name = session['info_register']['User_FirstName']
            last_name = session['info_register']['User_LastName']
            dob = session['info_register']['User_Dob']
            gender = session['info_register']['User_Gender']
            em = session['info_register']['User_Email']
            phone = session['info_register']['User_Phone']
            retyped_pwd = session['info_register']['User_PassWord']
            val = [first_name, last_name, dob, gender, em, phone, retyped_pwd, 2]
            res = obj_db.query_set(sql, val)
            if res:
                msg_text = "Resister new account successully! <a href='/signin'>Sign in now</a>"
                message.append(msg_text)
                session.clear()
                return jsonify(message=message)
            else:
                err_text = "Error in creating new account!"
                error.append(err_text)
                return jsonify(error=error) 
    return render_template('unlogedview/loginview/register.html')
