const firstName = document.getElementById("firstName");
const lastName = document.getElementById("lastName");
const birthdayDate = document.getElementById("birthdayDate");
const femaleGender = document.getElementById("femaleGender");
const maleGender = document.getElementById("maleGender");
const otherGender = document.getElementById("otherGender");
const emailAddress = document.getElementById("emailAddress");
const phoneNumber = document.getElementById("phoneNumber");
const password = document.getElementById("password");
const retypedPassword = document.getElementById("retypedPassword");

const verificationCode = document.getElementById("verificationCode");

const error_container = document.getElementById("error_container");
const message_container = document.getElementById("message_container");
const btn_submit_container = document.getElementById("btn_submit_container");
const email_verification_container = document.getElementById("email_verification_container");

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


function register() {
    fname = firstName.value;
    lname = lastName.value;
    dob = birthdayDate.value;
    f_gender = femaleGender.checked;
    m_gender = maleGender.checked;
    o_gender = otherGender.checked;
    email = emailAddress.value;
    phone = phoneNumber.value;
    pwd = password.value;
    retyped_pwd = retypedPassword.value;
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
        "phone": phone,
        "pwd": pwd,
        "retyped_pwd": retyped_pwd
    };
    input_data = JSON.stringify(input_data);

    const data = new FormData();
    data.append("input_register", input_data);
    $.ajax({
        url: "/register",
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
                error_container.innerHTML = str_err;
                error_container.classList.remove("d-none")
                error_container.classList.add("d-block")
                
                if (containsObject("d-none", message_container.classList) === false) {
                    message_container.classList.add("d-none")
                }
            }
            else if (typeof ret.message != "undefined") {
                msg = ret.message[0];
                if (msg === "Accepted input data") {
                    btn_submit_container.classList.remove("d-block");
                    btn_submit_container.classList.add("d-none");
                    email_verification_container.classList.remove("d-none");
                    email_verification_container.classList.add("d-block");
                }
                // str_msg = ""
                // ret.message.forEach(msg => {
                //     str_msg += "<p><i class='bi bi-check-circle-fill'></i> " + msg + "</p>";
                // });
                // message_container.innerHTML = str_msg;
                // message_container.classList.remove("d-none")
                // message_container.classList.add("d-block")

                if (containsObject("d-none", error_container.classList) === false) {
                    error_container.classList.add("d-none")
                }
            }
        },
    });
}


function verify() {
    input_code = verificationCode.value;
    console.log(input_code);
    input_data = {
        "input_code": input_code
    }

    input_data = JSON.stringify(input_data);

    const data = new FormData();
    data.append("email_verification", input_data);
    $.ajax({
        url: "/register",
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
                error_container.innerHTML = str_err;
                error_container.classList.remove("d-none")
                error_container.classList.add("d-block")
                
                if (containsObject("d-none", message_container.classList) === false) {
                    message_container.classList.add("d-none")
                }
                btn_submit_container.classList.remove("d-block");
                btn_submit_container.classList.add("d-none");
                email_verification_container.classList.remove("d-none");
                email_verification_container.classList.add("d-block");
            }
            else if (typeof ret.message != "undefined") {
                msg = ret.message[0];
                str_msg = ""
                ret.message.forEach(msg => {
                    str_msg += "<p><i class='bi bi-check-circle-fill'></i> " + msg + "</p>";
                });
                message_container.innerHTML = str_msg;
                message_container.classList.remove("d-none")
                message_container.classList.add("d-block")

                if (containsObject("d-none", error_container.classList) === false) {
                    error_container.classList.add("d-none")
                }
                btn_submit_container.classList.remove("d-block");
                btn_submit_container.classList.add("d-none");
                email_verification_container.classList.remove("d-block");
                email_verification_container.classList.add("d-none");
            }
        },
    });
}