# from fastapi import FastAPI, UploadFile, File
# from fastapi.responses import JSONResponse
# import cv2
# import numpy as np
# from insightface.app import FaceAnalysis
# from numpy.linalg import norm

# app = FastAPI()

# # 初始化模型，只加载一次
# recognizer = FaceAnalysis(providers=['CPUExecutionProvider'])
# recognizer.prepare(ctx_id=-1)

# def cos_similarity(emb1, emb2):
#     return float(emb1.dot(emb2) / (norm(emb1) * norm(emb2)))

# def extract_embedding(img_bytes):
#     img_array = np.frombuffer(img_bytes, np.uint8)
#     img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
#     faces = recognizer.get(img)
#     if len(faces) == 1:
#         return faces[0].embedding
#     return None

# @app.post("/compare/")
# async def compare_faces(input_image: UploadFile = File(...), db_image: UploadFile = File(...), threshold: float = 0.5):
#     input_bytes = await input_image.read()
#     db_bytes = await db_image.read()
#     emb1 = extract_embedding(input_bytes)
#     emb2 = extract_embedding(db_bytes)
#     if emb1 is None or emb2 is None:
#         return JSONResponse({"success": False, "reason": "face not detected"})
#     sim = cos_similarity(emb1, emb2)
#     return JSONResponse({
#         "success": True,
#         "similarity": sim,
#         "match": sim > threshold
#     })



# from fastapi import FastAPI, UploadFile, File
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import JSONResponse
# import face_recognition
# import shutil
# import os
# import uuid

# app = FastAPI()

# # CORS 中间件，允许前端访问
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # 你可以指定为前端地址如 http://localhost:3000
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# @app.post("/compare/")
# async def compare_faces(image: UploadFile = File(...)):
#     # 保存上传的图片
#     temp_filename = f"/tmp/{uuid.uuid4().hex}.jpg"
#     with open(temp_filename, "wb") as buffer:
#         shutil.copyfileobj(image.file, buffer)

#     try:
#         # 加载上传的图片
#         uploaded_image = face_recognition.load_image_file(temp_filename)
#         uploaded_encodings = face_recognition.face_encodings(uploaded_image)

#         if len(uploaded_encodings) == 0:
#             return JSONResponse({"message": "No face found"}, status_code=400)

#         # 模拟已有数据库中的参考图（可替换为本地文件）
#         known_image = face_recognition.load_image_file("reference.jpg")
#         known_encoding = face_recognition.face_encodings(known_image)[0]

#         # 只比较第一个识别出来的脸
#         match_result = face_recognition.compare_faces([known_encoding], uploaded_encodings[0])[0]

#         if match_result:
#             return {"message": "Face matched ✅"}
#         else:
#             return {"message": "Face not matched ❌"}
#     except Exception as e:
#         return JSONResponse({"message": f"Error processing image: {str(e)}"}, status_code=500)
#     finally:
#         # 清理临时文件
#         os.remove(temp_filename)


# from fastapi import FastAPI, UploadFile, File
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import JSONResponse
# import numpy as np
# import cv2
# import os
# from insightface.app import FaceAnalysis
# from numpy.linalg import norm

# app = FastAPI()

# # 允许跨域，方便本地或前端云部署访问
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # 生产可限制前端域名
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # 模型初始化，只加载一次（很快）
# recognizer = FaceAnalysis(providers=['CPUExecutionProvider'])
# recognizer.prepare(ctx_id=-1)

# DB_FOLDER = "db_images"  # 库文件夹，放你要比对的db图像

# def extract_embedding(img_bytes):
#     img_array = np.frombuffer(img_bytes, np.uint8)
#     img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
#     faces = recognizer.get(img)
#     if len(faces) == 1:
#         return faces[0].embedding
#     return None

# def cos_similarity(emb1, emb2):
#     return float(emb1.dot(emb2) / (norm(emb1) * norm(emb2)))

# @app.post("/compare/")
# async def compare_faces(input_image: UploadFile = File(...)):
#     # 读取上传图片
#     input_bytes = await input_image.read()
#     emb_input = extract_embedding(input_bytes)
#     if emb_input is None:
#         return JSONResponse({"success": False, "reason": "未检测到人脸"}, status_code=400)

#     # 遍历db_images文件夹，逐张比对
#     max_score = -1
#     matched_file = None
#     threshold = 0.5  # 阈值可自定义
#     for filename in os.listdir(DB_FOLDER):
#         if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
#             with open(os.path.join(DB_FOLDER, filename), "rb") as f:
#                 db_bytes = f.read()
#                 emb_db = extract_embedding(db_bytes)
#                 if emb_db is not None:
#                     sim = cos_similarity(emb_input, emb_db)
#                     if sim > max_score:
#                         max_score = sim
#                         matched_file = filename

#     # 返回最高分的库文件及是否match
#     return {
#         "success": True,
#         "matched_file": matched_file,
#         "similarity": max_score,
#         "match": max_score > threshold
#     }









# ----------------------------------------------------------------------------






# from fastapi import FastAPI, UploadFile, File, Form
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import JSONResponse
# import numpy as np
# import cv2
# import os
# from insightface.app import FaceAnalysis
# from numpy.linalg import norm
# # Google Cloud Storage


# app = FastAPI()

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# recognizer = FaceAnalysis(providers=['CPUExecutionProvider'])
# recognizer.prepare(ctx_id=-1)

# DB_FOLDER = "db_images"  # Student photo database

# def extract_embedding(img_bytes):
#     img_array = np.frombuffer(img_bytes, np.uint8)
#     img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
#     faces = recognizer.get(img)
#     if len(faces) == 1:
#         return faces[0].embedding
#     return None

# def cos_similarity(emb1, emb2):
#     return float(emb1.dot(emb2) / (norm(emb1) * norm(emb2)))

# @app.post("/compare/")
# async def compare_faces(
#     input_image: UploadFile = File(...),
#     buid: str = Form(...)
# ):
#     # Compose the path of the reference image based on BUID
#     filename = f"{buid}.jpg"
#     db_path = os.path.join(DB_FOLDER, filename)
#     if not os.path.isfile(db_path):
#         return JSONResponse(
#             {"success": False, "reason": "No reference image found for the given BUID."},
#             status_code=404
#         )

#     # Read and extract embedding from the uploaded image
#     input_bytes = await input_image.read()
#     emb_input = extract_embedding(input_bytes)
#     if emb_input is None:
#         return JSONResponse(
#             {"success": False, "reason": "No face detected in the uploaded image. Please try again."},
#             status_code=400
#         )

#     # Read and extract embedding from the uploaded image
#     with open(db_path, "rb") as f:
#         db_bytes = f.read()
#         emb_db = extract_embedding(db_bytes)
#         if emb_db is None:
#             return JSONResponse(
#                 {"success": False, "reason": "No face detected in the reference image."},
#                 status_code=500
#             )

#     # Calculate similarity
#     sim = cos_similarity(emb_input, emb_db)
#     threshold = 0.5  

#     return {
#         "success": True,
#         "buid": buid,
#         "similarity": sim,
#         "match": sim > threshold,
#         "db_image": filename,
#         "message": "Not match, please contact with TA or Professor" if sim <= threshold else "you are the one here! ✅"
#     }

# -------------------------------------------------------------------------------------

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import cv2
import os
from insightface.app import FaceAnalysis
from numpy.linalg import norm
# 🔴 新增：导入 Google Cloud Storage 相关库
from google.cloud import storage
from google.api_core import exceptions

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

recognizer = FaceAnalysis(providers=['CPUExecutionProvider'])
recognizer.prepare(ctx_id=-1)

# 🔴 修改：原来的本地文件夹配置，现在改为 Cloud Storage 配置
# DB_FOLDER = "db_images"  # 删除这行，不再使用本地文件夹
BUCKET_NAME = "face-identify-bucket"  # 🔴 新增：你的 Cloud Storage 存储桶名称
DB_FOLDER_PREFIX = "db_images/"       # 🔴 新增：存储桶中的文件夹前缀
# 🔴 新增：支持的图片格式列表
SUPPORTED_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']

# 🔴 新增：初始化 Cloud Storage 客户端，用于连接和操作存储桶
storage_client = storage.Client()

# # 🔴 新增：从 Cloud Storage 下载参考图片的函数
# def download_reference_image(buid):
#     """从 Cloud Storage 下载参考图片，替换原来的本地文件读取"""
#     try:
#         bucket = storage_client.bucket(BUCKET_NAME)
#         blob_name = f"{DB_FOLDER_PREFIX}{buid}.jpg"
#         blob = bucket.blob(blob_name)
        
#         # 检查文件是否存在（替换原来的 os.path.isfile 检查）
#         if not blob.exists():
#             return None
            
#         # 下载图片数据（替换原来的 open() 文件读取）
#         img_bytes = blob.download_as_bytes()
#         return img_bytes
        
#     except exceptions.NotFound:
#         return None
#     except Exception as e:
#         print(f"Error downloading image: {e}")
#         return None


# 🔴 修改：支持多种格式的图片下载函数
def download_reference_image(buid):
    """从 Cloud Storage 下载参考图片，支持多种格式"""
    try:
        bucket = storage_client.bucket(BUCKET_NAME)
        
        # 🔴 新逻辑：尝试多种文件格式
        for format_ext in SUPPORTED_FORMATS:
            blob_name = f"{DB_FOLDER_PREFIX}{buid}{format_ext}"
            blob = bucket.blob(blob_name)
            
            if blob.exists():
                print(f"Found image: {blob_name}")  # 调试信息
                img_bytes = blob.download_as_bytes()
                return img_bytes, format_ext  # 🔴 返回格式信息
        
        # 如果所有格式都找不到
        print(f"No image found for BUID: {buid}")
        return None, None
        
    except exceptions.NotFound:
        return None, None
    except Exception as e:
        print(f"Error downloading image: {e}")
        return None, None

# def extract_embedding(img_bytes):
#     """提取人脸特征 - 这个函数保持不变"""
#     img_array = np.frombuffer(img_bytes, np.uint8)
#     img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
#     faces = recognizer.get(img)
#     if len(faces) == 1:
#         return faces[0].embedding
#     return None

def extract_embedding(img_bytes):
    """提取人脸特征 - 支持所有格式"""
    img_array = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    faces = recognizer.get(img)
    if len(faces) == 1:
        return faces[0].embedding
    return None

def cos_similarity(emb1, emb2):
    """计算相似度 - 这个函数保持不变"""
    return float(emb1.dot(emb2) / (norm(emb1) * norm(emb2)))

# @app.post("/compare/")
# async def compare_faces(
#     input_image: UploadFile = File(...),
#     buid: str = Form(...)
# ):
#     # 🔴 修改：使用 Cloud Storage 下载参考图片，替换原来的本地文件路径检查
#     # 原来的代码：
#     # filename = f"{buid}.jpg"
#     # db_path = os.path.join(DB_FOLDER, filename)
#     # if not os.path.isfile(db_path):
    
#     db_bytes = download_reference_image(buid)
#     if db_bytes is None:
#         return JSONResponse(
#             {"success": False, "reason": "No reference image found for the given BUID."},
#             status_code=404
#         )

#     # 🔴 保持不变：读取上传图片的处理逻辑
#     input_bytes = await input_image.read()
#     emb_input = extract_embedding(input_bytes)
#     if emb_input is None:
#         return JSONResponse(
#             {"success": False, "reason": "No face detected in the uploaded image. Please try again."},
#             status_code=400
#         )

#     # 🔴 修改：直接使用从 Cloud Storage 下载的图片数据，替换原来的文件读取
#     # 原来的代码：
#     # with open(db_path, "rb") as f:
#     #     db_bytes = f.read()
#     #     emb_db = extract_embedding(db_bytes)
    
#     emb_db = extract_embedding(db_bytes)
#     if emb_db is None:
#         return JSONResponse(
#             {"success": False, "reason": "No face detected in the reference image."},
#             status_code=500
#         )

#     # 🔴 保持不变：相似度计算和返回结果的逻辑
#     sim = cos_similarity(emb_input, emb_db)
#     threshold = 0.5

#     return {
#         "success": True,
#         "buid": buid,
#         "similarity": sim,
#         "match": sim > threshold,
#         "db_image": f"{buid}.jpg",  # 🔴 小修改：直接使用 buid 构造文件名
#         "message": "Not match, please contact with TA or Professor" if sim <= threshold else "you are the one here! ✅"
#     }

# # 🔴 新增：可选的图片管理功能，让你可以通过 API 上传新的参考图片
# @app.post("/upload_reference/")
# async def upload_reference_image(
#     buid: str = Form(...),
#     image: UploadFile = File(...)
# ):
#     """上传新的参考图片到 Cloud Storage，不需要重新部署服务"""
#     try:
#         # 验证图片中是否有人脸
#         img_bytes = await image.read()
#         embedding = extract_embedding(img_bytes)
#         if embedding is None:
#             return JSONResponse(
#                 {"success": False, "reason": "No face detected in the uploaded image."},
#                 status_code=400
#             )
        
#         # 上传到 Cloud Storage
#         bucket = storage_client.bucket(BUCKET_NAME)
#         blob_name = f"{DB_FOLDER_PREFIX}{buid}.jpg"
#         blob = bucket.blob(blob_name)
        
#         # 重置文件指针并上传
#         await image.seek(0)
#         img_bytes = await image.read()
#         blob.upload_from_string(img_bytes, content_type='image/jpeg')
        
#         return {
#             "success": True,
#             "message": f"Reference image for BUID {buid} uploaded successfully.",
#             "blob_name": blob_name
#         }
        
#     except Exception as e:
#         return JSONResponse(
#             {"success": False, "reason": f"Upload failed: {str(e)}"},
#             status_code=500
#         )

# # 🔴 新增：删除参考图片的功能
# @app.delete("/delete_reference/{buid}")
# async def delete_reference_image(buid: str):
#     """删除参考图片"""
#     try:
#         bucket = storage_client.bucket(BUCKET_NAME)
#         blob_name = f"{DB_FOLDER_PREFIX}{buid}.jpg"
#         blob = bucket.blob(blob_name)
        
#         if not blob.exists():
#             return JSONResponse(
#                 {"success": False, "reason": "Reference image not found."},
#                 status_code=404
#             )
        
#         blob.delete()
#         return {
#             "success": True,
#             "message": f"Reference image for BUID {buid} deleted successfully."
#         }
        
#     except Exception as e:
#         return JSONResponse(
#             {"success": False, "reason": f"Delete failed: {str(e)}"},
#             status_code=500
#         )

# # 🔴 新增：列出所有参考图片的功能
# @app.get("/list_references/")
# async def list_reference_images():
#     """列出所有参考图片，方便管理"""
#     try:
#         bucket = storage_client.bucket(BUCKET_NAME)
#         blobs = bucket.list_blobs(prefix=DB_FOLDER_PREFIX)
        
#         images = []
#         for blob in blobs:
#             if blob.name.endswith('.jpg'):
#                 buid = blob.name.replace(DB_FOLDER_PREFIX, '').replace('.jpg', '')
#                 images.append({
#                     "buid": buid,
#                     "filename": blob.name,
#                     "size": blob.size,
#                     "updated": blob.updated.isoformat() if blob.updated else None
#                 })
        
#         return {
#             "success": True,
#             "images": images,
#             "total": len(images)
#         }
        
#     except Exception as e:
#         return JSONResponse(
#             {"success": False, "reason": f"List failed: {str(e)}"},
#             status_code=500
#         )




@app.post("/compare/")
async def compare_faces(
    input_image: UploadFile = File(...),
    buid: str = Form(...)
):
    # 🔴 修改：使用新的多格式下载函数
    db_bytes, found_format = download_reference_image(buid)
    if db_bytes is None:
        return JSONResponse(
            {"success": False, "reason": "No reference image found for the given BUID."},
            status_code=404
        )

    # 读取上传图片
    input_bytes = await input_image.read()
    emb_input = extract_embedding(input_bytes)
    if emb_input is None:
        return JSONResponse(
            {"success": False, "reason": "No face detected in the uploaded image. Please try again."},
            status_code=400
        )

    # 提取参考图片特征
    emb_db = extract_embedding(db_bytes)
    if emb_db is None:
        return JSONResponse(
            {"success": False, "reason": "No face detected in the reference image."},
            status_code=500
        )

    # 计算相似度
    sim = cos_similarity(emb_input, emb_db)
    threshold = 0.5

    return {
        "success": True,
        "buid": buid,
        "similarity": sim,
        "match": sim > threshold,
        "db_image": f"{buid}{found_format}",  # 🔴 显示实际找到的格式
        "format": found_format.replace('.', ''),  # 🔴 新增：显示图片格式
        "message": "Not match, please contact with TA or Professor" if sim <= threshold else "you are the one here! ✅"
    }

# 🔴 修改：支持多格式的图片上传
@app.post("/upload_reference/")
async def upload_reference_image(
    buid: str = Form(...),
    image: UploadFile = File(...),
    format: str = Form("jpg")  # 🔴 新增：可选择保存格式
):
    """上传新的参考图片到 Cloud Storage，支持多种格式"""
    try:
        # 验证图片中是否有人脸
        img_bytes = await image.read()
        embedding = extract_embedding(img_bytes)
        if embedding is None:
            return JSONResponse(
                {"success": False, "reason": "No face detected in the uploaded image."},
                status_code=400
            )
        
        # 🔴 验证格式参数
        if not format.startswith('.'):
            format = f".{format}"
        
        if format not in SUPPORTED_FORMATS:
            return JSONResponse(
                {"success": False, "reason": f"Unsupported format. Supported: {', '.join(SUPPORTED_FORMATS)}"},
                status_code=400
            )
        
        # 上传到 Cloud Storage
        bucket = storage_client.bucket(BUCKET_NAME)
        blob_name = f"{DB_FOLDER_PREFIX}{buid}{format}"
        blob = bucket.blob(blob_name)
        
        # 设置正确的 Content-Type
        content_type_map = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg', 
            '.png': 'image/png',
            '.bmp': 'image/bmp',
            '.tiff': 'image/tiff',
            '.webp': 'image/webp'
        }
        
        # 重置文件指针并上传
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

# 🔴 修改：支持多格式的删除功能
@app.delete("/delete_reference/{buid}")
async def delete_reference_image(buid: str):
    """删除参考图片（自动检测格式）"""
    try:
        bucket = storage_client.bucket(BUCKET_NAME)
        deleted_files = []
        
        # 🔴 新逻辑：查找并删除所有格式的文件
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

# 🔴 修改：支持多格式的列表显示
@app.get("/list_references/")
async def list_reference_images():
    """列出所有参考图片，支持多种格式"""
    try:
        bucket = storage_client.bucket(BUCKET_NAME)
        blobs = bucket.list_blobs(prefix=DB_FOLDER_PREFIX)
        
        images = []
        for blob in blobs:
            # 🔴 新逻辑：检查所有支持的格式
            for format_ext in SUPPORTED_FORMATS:
                if blob.name.endswith(format_ext):
                    blob.reload()
                    buid = blob.name.replace(DB_FOLDER_PREFIX, '').replace(format_ext, '')
                    images.append({
                        "buid": buid,
                        "filename": blob.name,
                        "format": format_ext.replace('.', ''),  # 🔴 新增：显示格式
                        "size": blob.size,
                        "updated": blob.updated.isoformat() if blob.updated else None
                    })
                    break  # 找到匹配的格式就停止
        
        return {
            "success": True,
            "images": images,
            "total": len(images),
            "supported_formats": [f.replace('.', '') for f in SUPPORTED_FORMATS]  # 🔴 新增：显示支持的格式
        }
        
    except Exception as e:
        return JSONResponse(
            {"success": False, "reason": f"List failed: {str(e)}"},
            status_code=500
        )

# 🔴 新增：获取特定学生的图片信息
@app.get("/get_reference_info/{buid}")
async def get_reference_info(buid: str):
    """获取特定学生的参考图片信息"""
    try:
        bucket = storage_client.bucket(BUCKET_NAME)
        found_images = []
        
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