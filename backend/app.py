import os
from io import BytesIO
from typing import Tuple

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as T
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# -- Model definition --
class MC_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.column1 = nn.Sequential(
            nn.Conv2d(3, 8, 9, padding=4), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 7, padding=3), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 7, padding=3), nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 7, padding=3), nn.ReLU(inplace=True),
            nn.Conv2d(16, 8, 7, padding=3), nn.ReLU(inplace=True)
        )
        self.column2 = nn.Sequential(
            nn.Conv2d(3, 10, 7, padding=3), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(10, 20, 5, padding=2), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(20, 40, 5, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(40, 20, 5, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(20, 10, 5, padding=2), nn.ReLU(inplace=True)
        )
        self.column3 = nn.Sequential(
            nn.Conv2d(3, 12, 5, padding=2), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(12, 24, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 48, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(48, 24, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(24, 12, 3, padding=1), nn.ReLU(inplace=True)
        )
        self.fusion_layer = nn.Sequential(
            nn.Conv2d(8 + 10 + 12, 1, kernel_size=1)
        )
    def forward(self, x):
        o1 = self.column1(x)
        o2 = self.column2(x)
        o3 = self.column3(x)
        cat = torch.cat([o1, o2, o3], dim=1)
        out = self.fusion_layer(cat)
        return out

def load_checkpoint_safe(path: str, model: nn.Module, device: torch.device):
    ckpt = torch.load(path, map_location=device)
    norm_state = {k.replace("module.", "", 1) if k.startswith("module.") else k: v for k, v in ckpt.items()}
    model.load_state_dict(norm_state, strict=False)
    model.to(device)
    model.eval()
    return model

def bytes_to_pil_image(data: bytes) -> Image.Image:
    try:
        pil = Image.open(BytesIO(data))
        pil.load()
        return pil.convert("RGB")
    except Exception:
        try:
            arr = np.frombuffer(data, dtype=np.uint8)
            img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img_bgr is not None:
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                return Image.fromarray(img_rgb).convert("RGB")
        except Exception:
            pass
        raise HTTPException(status_code=400, detail="Invalid/unsupported image format. Make sure you upload a real JPG/PNG image.")

def preprocess_pil_image(pil_image: Image.Image, size: Tuple[int, int]) -> torch.Tensor:
    return T.Compose([T.Resize(size), T.ToTensor()])(pil_image).unsqueeze(0)

# -- FastAPI setup --
CHECKPOINT_PATH = "./crowd_photo.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = (512, 512)
ALLOWED_ORIGINS = [
    "http://localhost:3000", "http://127.0.0.1:3000", "http://localhost", "http://127.0.0.1", "*"
]

model = MC_CNN()
model = load_checkpoint_safe(CHECKPOINT_PATH, model, DEVICE)

# app = FastAPI()
app = FastAPI(title="Photo Crowdcount API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/crowdcount")
async def crowd_count(image: UploadFile = File(...)):
    data = await image.read()
    if not data or len(data) < 100:  # quick sanity check: avoid empty/broken uploads
        raise HTTPException(status_code=400, detail="No image uploaded or file is too small.")
    pil_img = bytes_to_pil_image(data)
    inp = preprocess_pil_image(pil_img, IMAGE_SIZE).to(DEVICE)
    with torch.inference_mode():
        out = model(inp)
        dmap = out.detach().cpu()
        if dmap.ndim == 4:
            if dmap.shape[1] == 1: dmap = dmap.squeeze(1)
            else: dmap = dmap.sum(dim=1)
        dmap = dmap.squeeze(0).numpy()
    orig_w, orig_h = pil_img.size
    try:
        dmap_resized = cv2.resize(dmap, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    except Exception:
        dmap_resized = dmap
    count_int = int(round(dmap_resized.sum()))
    return {"count": count_int}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
