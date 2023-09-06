from fastapi import FastAPI, UploadFile
from fastapi.responses import JSONResponse
import base64

import numpy as np
import cv2

from utils.image_processing import detect_qr_code, extract_qr_code, post_processing

app = FastAPI(
    title="QR Code Processor API",
    description="An API for uploading an image, extracting QR codes.",
    version="1.0.0",
)


@app.post("/upload/")
async def upload_file(file: UploadFile):
    """
    Endpoint to upload an image file and extract a QR code from it.

    Args:
        file (UploadFile): The image file uploaded by the user.

    Returns:
        JSONResponse: A JSON response containing the extracted QR code as an image, the data
        decoded from the QR code, and a new QR code.
    """
    # Check if the uploaded file is an image
    if not file.content_type.startswith("image/"):
        return JSONResponse(content={"error": "No image file"}, status_code=400)

    try:
        # Read the image data from the uploaded file
        image_bytes = await file.read()

        # Convert the image bytes to a NumPy array
        image_np = np.frombuffer(image_bytes, np.uint8)

        # Decode the image using OpenCV
        image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

        # Detect the QR code in the image
        qr_code_normal = detect_qr_code(image)

        # Extract the QR code
        qr_code = extract_qr_code(qr_code_normal)

        # Post processing
        output_qr = post_processing(qr_code)

        # Decode the QR code using pyzbar
        decoder = cv2.QRCodeDetector()
        data, _, _ = decoder.detectAndDecode(output_qr)

        # Encode the extracted QR code as a JPEG image
        _, buffer = cv2.imencode(".jpg", output_qr)

        # Encode the image buffer as base64 and decode it to a UTF-8 string
        result = base64.b64encode(buffer).decode("utf-8")

        # Return the base64-encoded QR code image along with the decoded data and the new QR code
        return JSONResponse(content={"text": data, "qr_code": result})

    except Exception as e:
        # Handle any exceptions and return an error response
        return JSONResponse(content={"error": str(e)}, status_code=500)
