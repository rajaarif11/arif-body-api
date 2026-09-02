import os, uuid
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="AK BODY API", docs_url="/docs")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

os.makedirs("uploads", exist_ok=True)
os.makedirs("outputs", exist_ok=True)
os.makedirs("templates/male", exist_ok=True)
os.makedirs("templates/female", exist_ok=True)

class GenerateRequest(BaseModel):
    image: str
    category: str = "male"
    template: str = "auto"

def run_composite(user_img_path: str, template_path: str, output_path: str):
    user_img = Image.open(user_img_path).convert("RGBA")
    tpl_img = Image.open(template_path).convert("RGBA")
    t_w, t_h = tpl_img.size
    u_w, u_h = user_img.size
    scale = (t_w * 0.35) / max(int(u_w * 0.45), 1)
    new_w, new_h = int(u_w * scale), int(u_h * scale)
    user_resized = user_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", (t_w, t_h), (255, 255, 255, 0))
    offset_x = (t_w // 2) - (new_w // 2)
    offset_y = int(t_h * 0.10)
    canvas.paste(user_resized, (offset_x, offset_y), mask=user_resized.split()[3])
    final_output = Image.alpha_composite(canvas, tpl_img)
    final_output.save(output_path, "PNG", optimize=True)

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
    templates = [f for f in os.listdir(folder) if f.endswith(".png")]
    if not templates:
        raise HTTPException(status_code=404, detail="No template available")
    chosen = templates[0] if data.template == "auto" else f"{data.template}.png"
    tpl_path = os.path.join(folder, chosen)
    out_file = f"AK-{uuid.uuid4().hex[:6].upper()}.png"
    out_path = os.path.join("outputs", out_file)
    run_composite(input_path, tpl_path, out_path)
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
