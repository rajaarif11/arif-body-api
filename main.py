import os, uuid
import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="AK BODY API", docs_url="/docs")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("uploads", exist_ok=True)
os.makedirs("outputs", exist_ok=True)
os.makedirs("templates/male", exist_ok=True)
os.makedirs("templates/female", exist_ok=True)

# প্রাক-প্রশিক্ষিত ফেস ডিটেক্টর লোড
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

class GenerateRequest(BaseModel):
    image: str
    category: str = "male"
    template: str = "auto"

def auto_align_composite(user_img_path: str, template_path: str, output_path: str):
    # ইউজার ছবি ওপেন ও কালার কনভার্ট
    user_img_bgr = cv2.imread(user_img_path)
    if user_img_bgr is None:
        raise HTTPException(status_code=400, detail="Invalid image file")
    
    gray = cv2.cvtColor(user_img_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    
    user_pil = Image.open(user_img_path).convert("RGBA")
    tpl_pil = Image.open(template_path).convert("RGBA")
    
    u_w, u_h = user_pil.size
    t_w, t_h = tpl_pil.size

    if len(faces) > 0:
        # সবচেয়ে বড় মুখটি সিলেক্ট করা
        fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
        face_center_x = fx + (fw // 2)
        chin_y = fy + int(fh * 1.05)  # চিবুকের আনুমানিক অবস্থান
        
        # মুখের চওড়ার সাপেক্ষে স্যুটের স্কেলিং (সাধারণত কাঁধ মুখের ৩ গুণ চওড়া হয়)
        target_suit_w = int(fw * 3.2)
        scale_factor = target_suit_w / t_w
        target_suit_h = int(t_h * scale_factor)
        
        tpl_resized = tpl_pil.resize((target_suit_w, target_suit_h), Image.Resampling.LANCZOS)
        
        # স্যুটের কলার চিবুকের নিচে বসানোর জন্য অফসেট ক্যালকুলেশন
        suit_x = face_center_x - (target_suit_w // 2)
        # টেমপ্লেটের কলার সাধারণত টেমপ্লেটের টপ থেকে ৫-১০% নিচে থাকে
        suit_y = chin_y - int(target_suit_h * 0.08)
    else:
        # যদি কোনো ফেস ডিটেক্ট না হয় (ফলব্যাক স্কেল)
        scale_factor = u_w / t_w
        target_suit_w = u_w
        target_suit_h = int(t_h * scale_factor)
        tpl_resized = tpl_pil.resize((target_suit_w, target_suit_h), Image.Resampling.LANCZOS)
        suit_x = 0
        suit_y = int(u_h * 0.40)

    # ইউজারের মূল ক্যানভাসে স্যুটটি আলফা ব্লেন্ড করা
    canvas = user_pil.copy()
    canvas.paste(tpl_resized, (suit_x, suit_y), mask=tpl_resized.split()[3])
    canvas.save(output_path, "PNG", optimize=True)

@app.post("/api/v1/upload")
async def upload_file(file: UploadFile = File(...)):
    ext = file.filename.split(".")[-1].lower()
    if ext not in ["jpg", "jpeg", "png", "webp"]:
        raise HTTPException(status_code=400, detail="Invalid format")
    file_id = f"user_{uuid.uuid4().hex[:8]}.{ext}"
    path = os.path.join("uploads", file_id)
    with open(path, "wb") as f:
        f.write(await file.read())
    return {"success": True, "file_id": file_id}

@app.post("/api/v1/generate")
async def generate(data: GenerateRequest):
    input_path = os.path.join("uploads", data.image)
    if not os.path.exists(input_path):
        raise HTTPException(status_code=404, detail="Image not found")
    folder = os.path.join("templates", data.category.lower())
    if not os.path.exists(folder):
        raise HTTPException(status_code=404, detail="Category not found")
    templates = [f for f in os.listdir(folder) if f.endswith(".png")]
    if not templates:
        raise HTTPException(status_code=404, detail="No template available")
    
    chosen = templates[0] if data.template == "auto" else f"{data.template}.png"
    tpl_path = os.path.join(folder, chosen)
    
    out_file = f"AK-{uuid.uuid4().hex[:6].upper()}.png"
    out_path = os.path.join("outputs", out_file)
    
    auto_align_composite(input_path, tpl_path, out_path)
    return {"success": True, "download_url": f"/outputs/{out_file}"}

@app.get("/outputs/{filename}")
def get_output(filename: str):
    path = os.path.join("outputs", filename)
    if os.path.exists(path):
        return FileResponse(path)
    raise HTTPException(status_code=404, detail="File not found")

@app.get("/")
def home():
    return {"status": "online", "docs": "/docs"}
