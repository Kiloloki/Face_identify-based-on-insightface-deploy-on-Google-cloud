# -------------------------------------------------------------------------------------
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import cv2
import os
from insightface.app import FaceAnalysis
from numpy.linalg import norm
from google.cloud import storage
from google.api_core import exceptions

# Initialize FastAPI app instance
app = FastAPI()

# Enable Cross-Origin Resource Sharing (CORS) to allow requests from any domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize face recognition model using InsightFace.
# The model loads only once when the server starts.
recognizer = FaceAnalysis(providers=['CPUExecutionProvider'])
recognizer.prepare(ctx_id=-1)

# Name of the Google Cloud Storage bucket for storing reference images
BUCKET_NAME = "face-identify-bucket"

# The folder prefix within the bucket where reference images are stored
DB_FOLDER_PREFIX = "db_images/"

# List of supported image file formats for reference and input images
SUPPORTED_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']

# Instantiate Google Cloud Storage client for interacting with the storage bucket
storage_client = storage.Client()


def download_reference_image(buid):
    """
    Downloads the reference image of a user (by BUID) from Google Cloud Storage.
    Attempts all supported formats (e.g., jpg, png, webp) in priority order.

    Args:
        buid (str): The user's Boston University ID (BUID), used as the file key.

    Returns:
        tuple: (image_bytes, file_extension) if found, or (None, None) if not found.
    """
    try:
        bucket = storage_client.bucket(BUCKET_NAME)
        # Try each supported format for this BUID
        for format_ext in SUPPORTED_FORMATS:
            blob_name = f"{DB_FOLDER_PREFIX}{buid}{format_ext}"
            blob = bucket.blob(blob_name)
            if blob.exists():
                print(f"Found image: {blob_name}")
                img_bytes = blob.download_as_bytes()
                return img_bytes, format_ext
        # No image found for any format
        print(f"No image found for BUID: {buid}")
        return None, None
    except exceptions.NotFound:
        # Bucket or blob not found
        return None, None
    except Exception as e:
        print(f"Error downloading image: {e}")
        return None, None


def extract_embedding(img_bytes):
    """
    Extracts facial embedding vector from an image byte stream.

    Args:
        img_bytes (bytes): Raw image bytes (any supported format).

    Returns:
        np.ndarray: Embedding vector if exactly one face detected; otherwise None.
    """
    img_array = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    faces = recognizer.get(img)
    if len(faces) == 1:
        return faces[0].embedding
    return None


def cos_similarity(emb1, emb2):
    """
    Computes the cosine similarity between two embedding vectors.

    Args:
        emb1, emb2 (np.ndarray): Face embedding vectors.

    Returns:
        float: Cosine similarity (range: -1 to 1, where higher means more similar).
    """
    return float(emb1.dot(emb2) / (norm(emb1) * norm(emb2)))


@app.post("/compare/")
async def compare_faces(
    input_image: UploadFile = File(...),
    buid: str = Form(...)
):
    """
    Compares the uploaded input face image to the reference image stored for the given BUID.

    Args:
        input_image (UploadFile): The image uploaded by the user for comparison.
        buid (str): Boston University ID; identifies which reference image to use.

    Returns:
        JSON response: Success status, similarity score, match result, etc.
    """
    # Try to download the user's reference image from GCS (tries all supported formats)
    db_bytes, found_format = download_reference_image(buid)
    if db_bytes is None:
        return JSONResponse(
            {"success": False, "reason": "No reference image found for the given BUID."},
            status_code=404
        )

    # Read and process the uploaded input image
    input_bytes = await input_image.read()
    emb_input = extract_embedding(input_bytes)
    if emb_input is None:
        return JSONResponse(
            {"success": False, "reason": "No face detected in the uploaded image. Please try again."},
            status_code=400
        )

    # Process the reference image as well
    emb_db = extract_embedding(db_bytes)
    if emb_db is None:
        return JSONResponse(
            {"success": False, "reason": "No face detected in the reference image."},
            status_code=500
        )

    # Calculate similarity and determine if match
    sim = cos_similarity(emb_input, emb_db)
    threshold = 0.5  # Can be adjusted based on your use-case/empirical results

    return {
        "success": True,
        "buid": buid,
        "similarity": sim,
        "match": sim > threshold,
        "db_image": f"{buid}{found_format}",   # Actual filename in bucket
        "format": found_format.replace('.', ''),  # e.g., 'jpg', 'png'
        "message": (
            "Not match, please contact with TA or Professor"
            if sim <= threshold
            else "you are the one here! ✅"
        )
    }


@app.post("/upload_reference/")
async def upload_reference_image(
    buid: str = Form(...),
    image: UploadFile = File(...),
    format: str = Form("jpg")
):
    """
    Uploads a new reference face image to Google Cloud Storage for a given BUID.
    Supports multiple image formats (jpg, png, etc).

    Args:
        buid (str): Boston University ID; used as filename prefix.
        image (UploadFile): The uploaded image file (binary content).
        format (str): Image file extension/format (default 'jpg').

    Returns:
        JSON response: Success status, details, errors if any.
    """
    try:
        # Read and check the uploaded image
        img_bytes = await image.read()
        embedding = extract_embedding(img_bytes)
        if embedding is None:
            return JSONResponse(
                {"success": False, "reason": "No face detected in the uploaded image."},
                status_code=400
            )

        # Ensure format string starts with '.' (e.g., '.jpg')
        if not format.startswith('.'):
            format = f".{format}"

        # Reject unsupported formats
        if format not in SUPPORTED_FORMATS:
            return JSONResponse(
                {"success": False, "reason": f"Unsupported format. Supported: {', '.join(SUPPORTED_FORMATS)}"},
                status_code=400
            )

        # Prepare Cloud Storage objects
        bucket = storage_client.bucket(BUCKET_NAME)
        blob_name = f"{DB_FOLDER_PREFIX}{buid}{format}"
        blob = bucket.blob(blob_name)

        # Content-Type header mapping for different image formats
        content_type_map = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.bmp': 'image/bmp',
            '.tiff': 'image/tiff',
            '.webp': 'image/webp'
        }

        # Reset file pointer and upload
        await image.seek(0)
        img_bytes = await image.read()
        blob.upload_from_string(
            img_bytes,
            content_type=content_type_map.get(format, 'image/jpeg')
        )

        return {
            "success": True,
            "message": f"Reference image for BUID {buid} uploaded successfully.",
            "blob_name": blob_name,
            "format": format.replace('.', '')
        }

    except Exception as e:
        return JSONResponse(
            {"success": False, "reason": f"Upload failed: {str(e)}"},
            status_code=500
        )


@app.delete("/delete_reference/{buid}")
async def delete_reference_image(buid: str):
    """
    Deletes all reference images for a given BUID from Cloud Storage (across all supported formats).

    Args:
        buid (str): Boston University ID of the student.

    Returns:
        JSON response: List of deleted files or an error message if not found.
    """
    try:
        bucket = storage_client.bucket(BUCKET_NAME)
        deleted_files = []

        # Try to delete all formats for this BUID
        for format_ext in SUPPORTED_FORMATS:
            blob_name = f"{DB_FOLDER_PREFIX}{buid}{format_ext}"
            blob = bucket.blob(blob_name)
            if blob.exists():
                blob.delete()
                deleted_files.append(blob_name)

        if not deleted_files:
            return JSONResponse(
                {"success": False, "reason": "No reference image found for the given BUID."},
                status_code=404
            )

        return {
            "success": True,
            "message": f"Reference images for BUID {buid} deleted successfully.",
            "deleted_files": deleted_files
        }

    except Exception as e:
        return JSONResponse(
            {"success": False, "reason": f"Delete failed: {str(e)}"},
            status_code=500
        )


@app.get("/list_references/")
async def list_reference_images():
    """
    Lists all reference images currently stored in the Cloud Storage bucket.
    Supports all configured image formats.

    Returns:
        JSON response: Array of reference image metadata (BUID, filename, format, size, timestamp).
    """
    try:
        bucket = storage_client.bucket(BUCKET_NAME)
        blobs = bucket.list_blobs(prefix=DB_FOLDER_PREFIX)

        images = []
        for blob in blobs:
            # Check if the blob is in a supported image format
            for format_ext in SUPPORTED_FORMATS:
                if blob.name.endswith(format_ext):
                    blob.reload()
                    buid = blob.name.replace(DB_FOLDER_PREFIX, '').replace(format_ext, '')
                    images.append({
                        "buid": buid,
                        "filename": blob.name,
                        "format": format_ext.replace('.', ''),
                        "size": blob.size,
                        "updated": blob.updated.isoformat() if blob.updated else None
                    })
                    break  # Only one format match per file

        return {
            "success": True,
            "images": images,
            "total": len(images),
            "supported_formats": [f.replace('.', '') for f in SUPPORTED_FORMATS]
        }

    except Exception as e:
        return JSONResponse(
            {"success": False, "reason": f"List failed: {str(e)}"},
            status_code=500
        )


@app.get("/get_reference_info/{buid}")
async def get_reference_info(buid: str):
    """
    Returns all available reference image info for a specific BUID (one entry per format if exists).

    Args:
        buid (str): Boston University ID.

    Returns:
        JSON response: List of image files for the given BUID, with metadata.
    """
    try:
        bucket = storage_client.bucket(BUCKET_NAME)
        found_images = []

        # Try to find images of this BUID in all supported formats
        for format_ext in SUPPORTED_FORMATS:
            blob_name = f"{DB_FOLDER_PREFIX}{buid}{format_ext}"
            blob = bucket.blob(blob_name)
            if blob.exists():
                blob.reload()
                found_images.append({
                    "filename": blob_name,
                    "format": format_ext.replace('.', ''),
                    "size": blob.size,
                    "updated": blob.updated.isoformat() if blob.updated else None
                })

        if not found_images:
            return JSONResponse(
                {"success": False, "reason": "No reference images found for the given BUID."},
                status_code=404
            )

        return {
            "success": True,
            "buid": buid,
            "images": found_images,
            "total": len(found_images)
        }

    except Exception as e:
        return JSONResponse(
            {"success": False, "reason": f"Query failed: {str(e)}"},
            status_code=500
        )

