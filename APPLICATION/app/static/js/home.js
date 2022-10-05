function loadFile(event) {
    let blob = new Blob([event.target.files[0]], { type: "image/jpeg" });
    let fname = event.target.files[0].name;
    process(blob, fname, false);
}
  
function process(blob, fname, isDrop) {
    const image = document.getElementById("showUploadImage");
    const upload = document.getElementById("myUploadImage");
    const result = document.getElementById("result");
    const uploadview = document.getElementById("uploadview");
    let srcBlobImg = URL.createObjectURL(blob);
    image.src = srcBlobImg;
    upload.addEventListener("input", function () {
        original.classList.add("tab-active");
        gray.classList.remove("tab-active");
        new_gray.classList.remove("tab-active");
    });
    result.style.display = "block";
    // uploadview.style.paddingTop = "130px";
    result.scrollIntoView(true);
    let scrolledY = window.scrollY;

    if (scrolledY) {
        window.scroll(0, scrolledY - 130);
    }

    const original = document.getElementById("original");
    const gray = document.getElementById("gray");
    var $ = jQuery;
    // =================================================================================================================
    // Rgb
    // =================================================================================================================
    original.addEventListener("click", function (event) {
        original.classList.add("tab-active");
        new_gray.classList.remove("tab-active");
        image.src = srcBlobImg;
        event.preventDefault();
    });
    // =================================================================================================================
    // Rgb To Gray
    // =================================================================================================================
    const new_gray = gray.cloneNode(true);
    gray.parentNode.replaceChild(new_gray, gray);
    new_gray.addEventListener("click", function () {
        new_gray.classList.add("tab-active");
        original.classList.remove("tab-active");
        const data = new FormData();
        data.append("file", blob, fname);
        $.ajax({
        url: "/gray",
        type: "POST",
        processData: false,
        contentType: false,
        data: data,
        success: (ret) => {
            var srcNewImg = ret.data;
            $("#showUploadImage").attr("src", srcNewImg);
        },
        });
    });

    if (isDrop) {
        original.classList.add("tab-active");
        gray.classList.remove("tab-active");
        new_gray.classList.remove("tab-active");
    }
}

// download
function downloadImage() {
const download = document.getElementById("download");
const image = document.getElementById("showUploadImage");
download.href = image.src;
download.click();
}

// drag and drop
$(document).ready(function () {
    const dropContainer = document.getElementById("dropContainer");
    const error = document.getElementById("err");
    // console.log(dropContainer)
    dropContainer.ondragover = function (e) {
        e.preventDefault();
        dropContainer.style.border = "4px dashed green";
        return false;
    };

    dropContainer.ondragleave = function (e) {
        e.preventDefault();
        dropContainer.style.border = "3px dashed #4e7efe";
        return false;
    };

    dropContainer.ondrop = function (e) {
        e.preventDefault();
        dropContainer.style.border = "3px dashed #4e7efe";
        let link = e.dataTransfer.getData("text/html");
        let dropContext = $("<div>").append(link);
        let imgURL = $(dropContext).find("img").attr("src");
        if (imgURL) {
        fetch(imgURL)
            .then((res) => res.blob())
            .then((blob) => {
            error.style.display = "none";
            let index = imgURL.lastIndexOf("/") + 1;
            let filename = imgURL.substr(index);
            let allowedName = /(\.jpg|\.jpeg|\.png|\.gif)$/i;
            if (imgURL.includes("base64")) {
                error.innerText = "⚠️ Không thể kéo ảnh này, hãy mở nó ra rồi kéo";
                error.style.display = "block";
                return;
            }
            if (!allowedName.exec(filename)) {
                error.innerText =
                "⚠️ Không thể upload file này, vui lòng upload file khác";
                error.style.display = "block";
                return;
            }
            if (!filename.includes(".")) {
                error.innerText =
                "⚠️ Không thể upload file này, vui lòng upload file khác";
                error.style.display = "block";
                return;
            }
            process(blob, filename, true);
            })
            .catch(() => {
            error.innerText =
                "⚠️ Không thể upload file này, vui lòng upload file khác";
            error.style.display = "block";
            });
        } else {
        const file = e.dataTransfer.files[0];
        const fileType = file["type"];
        const validImageTypes = ["image/gif", "image/jpeg", "image/png"];
        if (!validImageTypes.includes(fileType)) {
            error.innerText =
            "⚠️ Không thể upload file này, vui lòng upload file khác";
            error.style.display = "block";
        } else {
            error.style.display = "none";
            let blob = new Blob([file], { type: "image/jpeg" });
            let fname = file.name;
            process(blob, fname, true);
        }
        }
    };
});


/* ============================================ */
/* Pagination */
/* ============================================ */
let pages = 54;

current_pathname = window.location.pathname // <path>/page/<int> or <path>

index_page = current_pathname.lastIndexOf("page")

current_page = ""
if (index_page < 0) { // not found "page" in URL
    current_page = 1
}
else {
    index_slash = current_pathname.lastIndexOf("/") // last found
    current_page = current_pathname.slice(index_slash+1)
    current_page = parseInt(current_page)
}


document.getElementById('pagination').innerHTML = createPagination(pages, current_page);

function createPagination(pages, page) {
    let str = '<ul>';
    let active;
    let pageCutLow = page - 1;
    let pageCutHigh = page + 1;
    // Show the Previous button only if you are on a page other than the first
    if (page > 1) {
        str += '<li class="page-item previous no"><a href="/page/' + String(page-1) +'" onclick="createPagination(pages, '+(page-1)+')">Previous</a></li>';
    }
    // Show all the pagination elements if there are less than 6 pages total
    if (pages < 6) {
        for (let p = 1; p <= pages; p++) {
            active = page == p ? "active" : "no";
            str += '<li class="'+active+'"><a href="/page/' + String(p) +'" onclick="createPagination(pages, '+p+')">'+ p +'</a></li>';
        }
    }
    // Use "..." to collapse pages outside of a certain range
    else {
    // Show the very first page followed by a "..." at the beginning of the
    // pagination section (after the Previous button)
        if (page > 2) {
            str += '<li class="no page-item"><a href="/page/' + String(1) +'" onclick="createPagination(pages, 1)">1</a></li>';
            if (page > 3) {
                str += '<li class="out-of-range"><a href="/page/' + String(page-2) +'" onclick="createPagination(pages,'+(page-2)+')">...</a></li>';
            }
        }
    // Determine how many pages to show after the current page index
        if (page === 1) {
            pageCutHigh += 2;
        } 
        else if (page === 2) {
            pageCutHigh += 1;
        }
    // Determine how many pages to show before the current page index
        if (page === pages) {
            pageCutLow -= 2;
        } else if (page === pages-1) {
            pageCutLow -= 1;
        }
    // Output the indexes for pages that fall inside the range of pageCutLow
    // and pageCutHigh
        for (let p = pageCutLow; p <= pageCutHigh; p++) {
            if (p === 0) {
                p += 1;
            }
            if (p > pages) {
                continue;
            }
            active = page == p ? "active" : "no";
            str += '<li class="page-item '+active+'"><a href="/page/' + String(p) +'" onclick="createPagination(pages, '+p+')">'+ p +'</a></li>';
        }
    // Show the very last page preceded by a "..." at the end of the pagination
    // section (before the Next button)
        if (page < pages-1) {
            if (page < pages-2) {
                str += '<li class="out-of-range"><a href="/page/' + String(page+2) +'" onclick="createPagination(pages,'+(page+2)+')">...</a></li>';
            }
            str += '<li class="page-item no"><a href="/page/' + String(pages) +'" onclick="createPagination(pages, pages)">'+pages+'</a></li>';
        }
    }
    // Show the Next button only if you are on a page other than the last
    if (page < pages) {
        str += '<li class="page-item next no"><a href="/page/' + String(page+1) +'" onclick="createPagination(pages, '+(page+1)+')">Next</a></li>';
    }
    str += '</ul>';
    // Return the pagination string to be outputted in the pug templates
    document.getElementById('pagination').innerHTML = str;
    return str;
}


/* ============================================ */
/* Scroll Back To Top Button */
/* ============================================ */
// Get the button:
let mybutton = document.getElementById("myBtn");

// When the user scrolls down 20px from the top of the document, show the button
window.onscroll = function() {scrollFunction()};

function scrollFunction() {
    if (document.body.scrollTop > 20 || document.documentElement.scrollTop > 20) {
        mybutton.style.display = "block";
    } else {
        mybutton.style.display = "none";
    }
}

// When the user clicks on the button, scroll to the top of the document
function topFunction() {
    document.body.scrollTop = 0; // For Safari
    document.documentElement.scrollTop = 0; // For Chrome, Firefox, IE and Opera
}

