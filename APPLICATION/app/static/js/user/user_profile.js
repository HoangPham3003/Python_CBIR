const firstName = document.getElementById("firstName");
const lastName = document.getElementById("lastName");
const birthdayDate = document.getElementById("birthdayDate");
const femaleGender = document.getElementById("femaleGender");
const maleGender = document.getElementById("maleGender");
const otherGender = document.getElementById("otherGender");
const emailAddress = document.getElementById("emailAddress");
const phoneNumber = document.getElementById("phoneNumber");
const currentPassword = document.getElementById("currentPassword");
const newPassword = document.getElementById("newPassword");
const retypedNewPassword = document.getElementById("retypedNewPassword");


const error_update_profile_container = document.getElementById("error_update_profile_container");
const message_update_profile_container = document.getElementById("message_update_profile_container");
const error_change_pwd_container = document.getElementById("error_change_pwd_container");
const message_change_pwd_container = document.getElementById("message_change_pwd_container");


const $ = jQuery;

function containsObject(obj, list) {
    var i;
    for (i = 0; i < list.length; i++) {
        if (list[i] === obj) {
            return true;
        }
    }
    return false;
}

function update_profile() {
    fname = firstName.value;
    lname = lastName.value;
    dob = birthdayDate.value;
    f_gender = femaleGender.checked;
    m_gender = maleGender.checked;
    o_gender = otherGender.checked;
    email = emailAddress.value;
    phone = phoneNumber.value;

    gender = 0
    if (f_gender === true) {
        gender = 0;
    }
    else if (m_gender === true) {
        gender = 1;
    }
    else if (o_gender === true) {
        gender = 2;
    }
    input_data = {
        "fname": fname,
        "lname": lname,
        "dob": dob,
        "gender": gender,
        "email": email,
        "phone": phone
    };
    input_data = JSON.stringify(input_data);

    const data = new FormData();
    data.append("input_update_profile", input_data);
    $.ajax({
        url: "/user/profile",
        type: "POST",
        processData: false,
        contentType: false,
        data: data,
        success: (ret) => {
            if (typeof ret.error != "undefined") {
                str_err = ""
                ret.error.forEach(err => {
                    str_err += "<p><i class='bi bi-exclamation-triangle-fill'></i> " + err + "</p>";
                });
                error_update_profile_container.innerHTML = str_err;
                error_update_profile_container.classList.remove("d-none")
                error_update_profile_container.classList.add("d-block")
                
                if (containsObject("d-none", message_update_profile_container.classList) === false) {
                    message_update_profile_container.classList.add("d-none")
                }
            }
            else if (typeof ret.message != "undefined") {
                str_msg = ""
                ret.message.forEach(msg => {
                    str_msg += "<p><i class='bi bi-check-circle-fill'></i> " + msg + "</p>";
                });
                message_update_profile_container.innerHTML = str_msg;
                message_update_profile_container.classList.remove("d-none")
                message_update_profile_container.classList.add("d-block")

                if (containsObject("d-none", error_update_profile_container.classList) === false) {
                    error_update_profile_container.classList.add("d-none")
                }

                // setTimeout(() => {  window.location.replace("/signout"); }, 1000);
            }
        },
    });
}


function change_pwd() {
    current_pwd = currentPassword.value;
    new_pwd = newPassword.value;
    retyped_new_pwd = retypedNewPassword.value;
    input_data = {
        "current_pwd": current_pwd,
        "new_pwd": new_pwd,
        "retyped_new_pwd": retyped_new_pwd
    };
    input_data = JSON.stringify(input_data);

    const data = new FormData();
    data.append("input_change_pwd", input_data);
    $.ajax({
        url: "/user/profile",
        type: "POST",
        processData: false,
        contentType: false,
        data: data,
        success: (ret) => {
            if (typeof ret.error != "undefined") {
                str_err = ""
                ret.error.forEach(err => {
                    str_err += "<p><i class='bi bi-exclamation-triangle-fill'></i> " + err + "</p>";
                });
                error_change_pwd_container.innerHTML = str_err;
                error_change_pwd_container.classList.remove("d-none")
                error_change_pwd_container.classList.add("d-block")
                
                if (containsObject("d-none", message_change_pwd_container.classList) === false) {
                    message_change_pwd_container.classList.add("d-none")
                }
            }
            else if (typeof ret.message != "undefined") {
                str_msg = ""
                msg_text = ""
                ret.message.forEach(msg => {
                    msg_text = msg
                    str_msg += "<p><i class='bi bi-check-circle-fill'></i> " + msg + "</p>";
                });
                message_change_pwd_container.innerHTML = str_msg;
                message_change_pwd_container.classList.remove("d-none")
                message_change_pwd_container.classList.add("d-block")

                if (containsObject("d-none", error_change_pwd_container.classList) === false) {
                    error_change_pwd_container.classList.add("d-none")
                }
                
                setTimeout(() => {  window.location.replace("/signout"); }, 1000);
            }
        },
    });
}