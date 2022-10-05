const emailAddress = document.getElementById("emailAddress");

const error_container = document.getElementById("error_container");
const message_container = document.getElementById("message_container");


function containsObject(obj, list) {
    var i;
    for (i = 0; i < list.length; i++) {
        if (list[i] === obj) {
            return true;
        }
    }
    return false;
}


function reset_password() {
    email = emailAddress.value;

    input_data = {
        "email": email
    }

    input_data = JSON.stringify(input_data);

    const data = new FormData();
    data.append("input_forgot_password", input_data);
    $.ajax({
        url: "/forgot-password",
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
            }
        },
    });
}