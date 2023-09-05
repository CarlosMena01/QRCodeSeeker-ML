document.getElementById("imageInput").addEventListener("change", function (event) {
    const file = event.target.files[0];
    if (file) {
        const formData = new FormData();
        formData.append("file", file);

        fetch("http://127.0.0.1:8000/upload/", {
            method: "POST",
            body: formData,
        })
        .then(response => response.json())
        .then(data => {
            document.getElementById("qrData").textContent = data.text;
            document.getElementById("originalQR").src = "data:image/jpeg;base64," + data.qr_code;
            document.getElementById("newQR").src = "data:image/jpeg;base64," + data.new_qr;
        })
        .catch(error => {
            console.error("Error:", error);
            alert("An error occurred.");
        });
    }
});

function uploadImage() {
    document.getElementById("imageInput").click();
}
