# Face Identify Service (FastAPI + InsightFace + Google Cloud Storage)

&#x20; &#x20;

A web service for face verification using InsightFace, FastAPI, and Google Cloud Storage.\
Supports multi-format reference images and secure student attendance with facial recognition.

---

## Features

- **Face Verification API**: Compare uploaded face images with student reference images (by BUID)
- **Multi-format Image Support**: Handles `.jpg`, `.png`, `.jpeg`, `.webp`, `.bmp`, `.tiff`
- **Cloud Storage**: All reference images are managed in a Google Cloud Storage bucket
- **RESTful Endpoints**: For uploading, comparing, deleting, and listing reference images
- **CORS Enabled**: Ready for frontend integration (React, Vue, etc.)
- **Robust Error Handling**: Clear, structured API responses for all failure modes

---

## Architecture Overview

```
+---------+       +-------------+        +-----------------+
|  User   | <---> |   Frontend  | <----> |   FastAPI App   |
+---------+       +-------------+        +-----------------+
                                             |
                                             v
                                +---------------------------+
                                | Google Cloud Storage (GCS)|
                                +---------------------------+
```

---

## Quick Start

### Prerequisites

- Python 3.8+
- Google Cloud project and [service account key](https://cloud.google.com/iam/docs/creating-managing-service-account-keys) (JSON)
- [InsightFace](https://github.com/deepinsight/insightface) & dependencies
- [Google Cloud Storage Python SDK](https://googleapis.dev/python/storage/latest/index.html)

### Installation

```bash
git clone https://github.com/Kiloloki/Face_identify-based-on-insightface-deploy-on-Google-cloud.git
cd Face_identify-based-on-insightface-deploy-on-Google-cloud

# (Recommended) Create and activate a Python virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set Google credentials (replace with your path)
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your-gcp-key.json"
```

### Running the Service

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

*Replace **``** with your actual Python file name if it's different.*

---

## API Documentation

FastAPI automatically provides interactive Swagger docs at\
`http://localhost:8000/docs`

### Main Endpoints

#### 1. Compare Faces

```http
POST /compare/
Form fields:
  - buid (str): Student's BUID
  - input_image (file): Photo to verify

Response:
  {
    "success": true,
    "buid": "U12345678",
    "similarity": 0.83,
    "match": true,
    "db_image": "U12345678.jpg",
    "format": "jpg",
    "message": "you are the one here! ✅"
  }
```

#### 2. Upload Reference Image

```http
POST /upload_reference/
Form fields:
  - buid (str): Student's BUID
  - image (file): Face photo to register
  - format (str): Image format (default: jpg)

Response:
  {
    "success": true,
    "message": "Reference image for BUID U12345678 uploaded successfully.",
    "blob_name": "db_images/U12345678.jpg",
    "format": "jpg"
  }
```

#### 3. Delete Reference Image

```http
DELETE /delete_reference/{buid}
Response:
  {
    "success": true,
    "message": "... deleted successfully.",
    "deleted_files": [...]
  }
```

#### 4. List All Reference Images

```http
GET /list_references/
Response:
  {
    "success": true,
    "images": [
      {
        "buid": "U12345678",
        "filename": "db_images/U12345678.jpg",
        "format": "jpg",
        "size": 12345,
        "updated": "2024-08-01T12:34:56.000Z"
      },
      ...
    ],
    "total": 23,
    "supported_formats": ["jpg", "jpeg", "png", "bmp", "tiff", "webp"]
  }
```

#### 5. Get Info for a Specific Student

```http
GET /get_reference_info/{buid}
Response:
  {
    "success": true,
    "buid": "U12345678",
    "images": [
      {
        "filename": "db_images/U12345678.jpg",
        "format": "jpg",
        "size": 12345,
        "updated": "2024-08-01T12:34:56.000Z"
      }
    ],
    "total": 1
  }
```

---

## Deployment Notes

- **Service Account**: Your Google credentials must have storage read/write permissions for the bucket.
- **Bucket Name**: Change the `BUCKET_NAME` in `main.py` as needed.
- **Firewall**: If deploying on cloud VM, open port 8000 (or your choice).
- **Scaling**: For production, use a process manager (e.g., Gunicorn) + reverse proxy (e.g., Nginx).

---

## TODO

- [ ] Add user authentication (JWT or OAuth)
- [ ] Optimize face embedding extraction for batch uploads
- [ ] Add more logging and monitoring
- [ ] Support image preview endpoints
- [ ] Write frontend integration sample


---

## License

MIT License (c) 2024 [Kiloloki](https://github.com/Kiloloki)

---

## Acknowledgements

- [InsightFace](https://github.com/deepinsight/insightface)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Google Cloud Storage Python Client](https://googleapis.dev/python/storage/latest/index.html)

