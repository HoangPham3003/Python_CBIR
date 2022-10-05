var menuHolder = document.getElementById("menuHolder")
function menuToggle() {
    if (menuHolder.className === "drawMenu") {
        menuHolder.className = "";
    }
    else {
        menuHolder.className = "drawMenu";
    }
}


// var list_categories = document.getElementById("list_categories");
// current_pathname = window.location.pathname;
// role_name = ""
// if (current_pathname.includes("user")) {
//     role_name = "user";
// }
// else if (current_pathname.includes("admin")) {
//     role_name = "admin"
// }
// var host = window.location.protocol + "//" + window.location.host + "/";
// if (role_name != "") {
//     for (let i = 0 ; i <  list_categories.children.length ; i++) {
//         li_tag = list_categories.children[i];
//         a_tag = li_tag.children[0];
//         href = a_tag.href;
//         index_images = href.indexOf("images");
//         new_href = host + role_name + "/" + href.slice(index_images);
//         list_categories.children[i].children[0].href = new_href;
//     }
// }