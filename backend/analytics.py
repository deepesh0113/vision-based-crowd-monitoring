import os
import time
import threading
import csv
from datetime import datetime
from io import StringIO
import tempfile

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse

# ---------------- SETTINGS ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "crowd_counting.pth")

USE_GPU = True
USE_FP16_IF_CUDA = True
FRAME_RESIZE = (512, 384)           # (width, height) - FIXED order
SKIP_FRAMES = 10
SMOOTH_WINDOW = 3
FLUSH_INTERVAL = 2.0
PRINT_EVERY = 50
UPSAMPLE_DMAP = False
SCALE_BY_AREA = True
# ------------------------------------------

device = torch.device("cuda" if (USE_GPU and torch.cuda.is_available()) else "cpu")
print("Device:", device)

# ---------- Camera hinderance helpers (EXACT COPY) ----------
_hinder_counters = {
    "frozen": 0,
    "covered": 0,
}
FROZEN_DIFF_THRESH = 2.0
FROZEN_FRAMES_THRESH = 5
COVERED_MEAN_THRESH = 15
COVERED_FRAMES_THRESH = 3
LOW_CONTRAST_STD_THRESH = 10.0
LOW_CONTRAST_FRAMES_THRESH = 5
DARK_PCT_THRESH = 0.50

_prev_gray_for_hinder = None

def check_camera_hinder(frame_bgr):  # EXACT COPY FROM WORKING VERSION
    """
    Inspect frame and update internal counters. Returns (is_hindered: bool, reason: str).
    """
    global _prev_gray_for_hinder, _hinder_counters

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    mean = float(gray.mean())
    std = float(gray.std())

    diff_mean = None
    if _prev_gray_for_hinder is None:
        diff_mean = 255.0
    else:
        diff = cv2.absdiff(gray, _prev_gray_for_hinder)
        diff_mean = float(diff.mean())

    edges = cv2.Canny(gray, 50, 150)

    if diff_mean < FROZEN_DIFF_THRESH:
        _hinder_counters["frozen"] += 1
    else:
        _hinder_counters["frozen"] = 0

    if mean < COVERED_MEAN_THRESH and std < LOW_CONTRAST_STD_THRESH:
        _hinder_counters["covered"] += 1
    else:
        _hinder_counters["covered"] = 0

    reason = ""
    if _hinder_counters["frozen"] >= FROZEN_FRAMES_THRESH:
        reason = "camera_frozen"
    elif _hinder_counters["covered"] >= COVERED_FRAMES_THRESH:
        reason = "lens_covered_or_extremely_dark"

    _prev_gray_for_hinder = gray.copy()
    return (bool(reason), reason)

def enhance_frame(frame_bgr, gamma=1.6, use_clahe=True, clahe_clip=2.0, clahe_tile=(8,8), denoise=False):
    img = frame_bgr.copy()
    if gamma != 1.0:
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(256)]).astype("uint8")
        img = cv2.LUT(img, table)

    if use_clahe:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_tile)
        v_clahe = clahe.apply(v)
        hsv_clahe = cv2.merge((h, s, v_clahe))
        img = cv2.cvtColor(hsv_clahe, cv2.COLOR_HSV2BGR)

    if denoise:
        img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

    img = np.clip(img, 0, 255).astype(np.uint8)
    return img

# ---------- MCNN model (EXACT SAME) ----------
class MC_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.column1 = nn.Sequential(
            nn.Conv2d(3, 8, 9, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 7, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 7, padding='same'),
            nn.ReLU(),
            nn.Conv2d(32, 16, 7, padding='same'),
            nn.ReLU(),
            nn.Conv2d(16, 8, 7, padding='same'),
            nn.ReLU(),
        )
        self.column2 = nn.Sequential(
            nn.Conv2d(3, 10, 7, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(10, 20, 5, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(20, 40, 5, padding='same'),
            nn.ReLU(),
            nn.Conv2d(40, 20, 5, padding='same'),
            nn.ReLU(),
            nn.Conv2d(20, 10, 5, padding='same'),
            nn.ReLU(),
        )
        self.column3 = nn.Sequential(
            nn.Conv2d(3, 12, 5, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(12, 24, 3, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 48, 3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(48, 24, 3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(24, 12, 3, padding='same'),
            nn.ReLU(),
        )
        self.fusion_layer = nn.Sequential(
            nn.Conv2d(30, 1, 1, padding=0),
        )

    def forward(self, img_tensor):
        x1 = self.column1(img_tensor)
        x2 = self.column2(img_tensor)
        x3 = self.column3(img_tensor)
        x = torch.cat((x1, x2, x3), 1)
        x = self.fusion_layer(x)
        return x

def load_model(path, device):
    m = MC_CNN().to(device).eval()
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        state = ckpt['state_dict']
    elif isinstance(ckpt, dict):
        state = ckpt
    elif isinstance(ckpt, nn.Module):
        ckpt.to(device).eval()
        return ckpt
    else:
        raise RuntimeError("Unsupported checkpoint format")
    
    new_state = {}
    for k, v in state.items():
        nk = k[len('module.'):] if k.startswith('module.') else k
        new_state[nk] = v
    
    missing, unexpected = m.load_state_dict(new_state, strict=False)
    if missing or unexpected:
        print("load_state_dict warnings:")
        if missing: print(" missing:", missing[:6], "...")
        if unexpected: print(" unexpected:", list(unexpected)[:6], "...")
    return m

model = load_model(MODEL_PATH, device)
if device.type == 'cuda' and USE_FP16_IF_CUDA:
    model.half()

# FIXED: Exact copy of working frame_to_tensor
def frame_to_tensor(frame_bgr):
    if FRAME_RESIZE is not None:
        frame_bgr = cv2.resize(frame_bgr, FRAME_RESIZE, interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    arr = np.asarray(rgb, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2,0,1))
    tensor = torch.from_numpy(arr).unsqueeze(0)
    if device.type == 'cuda' and USE_FP16_IF_CUDA:
        tensor = tensor.half().to(device, non_blocking=True)
    else:
        tensor = tensor.to(device, non_blocking=True)
    return tensor

def ms_to_time_str(t_ms):
    ms = int(t_ms % 1000)
    total_seconds = int(t_ms // 1000)
    minutes = total_seconds // 60
    seconds = total_seconds % 60
    return f"{minutes:02d}:{seconds:02d}:{ms:03d}"

# ---------- per-run tracking ----------
RUNS = {}
RUNS_LOCK = threading.Lock()

def process_video_to_csv(video_path: str, run_id: str):
    """
    FIXED: Exact logic from working version
    """
    global _hinder_counters, _prev_gray_for_hinder
    _hinder_counters = {k: 0 for k in _hinder_counters}
    _prev_gray_for_hinder = None

    buf = StringIO()
    writer = csv.writer(buf)
    today_str = datetime.now().strftime("%Y-%m-%d")
    writer.writerow(["date", "timestamp_ms", "count", "alert"])

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Cannot open video:", video_path)
        with RUNS_LOCK:
            RUNS[run_id]["csv"] = buf.getvalue()
            RUNS[run_id]["done"] = True
        return

    smoother = []
    processed = 0
    last_flush = time.time()
    start_wall = time.time()

    with torch.no_grad():
        while True:
            ok, frame = cap.read()
            if not ok:
                print("End of video reached.")
                break

            if processed % SKIP_FRAMES != 0:
                processed += 1
                continue

            # enhance and prepare
            enhanced = enhance_frame(frame, gamma=1.6, use_clahe=True, denoise=False)
            inp = frame_to_tensor(enhanced)

            # forward
            out = model(inp)

            # FIXED: Exact fp16 handling from working version
            if device.type == 'cuda' and USE_FP16_IF_CUDA:
                out_f = out.float()
            else:
                out_f = out

            # FIXED: Safety ReLU
            out_f = torch.relu(out_f)

            # FIXED: CRITICAL - get dmap on CPU as float32 (was missing!)
            if device.type == 'cuda' and USE_FP16_IF_CUDA:
                dmap = out_f.float().squeeze(0).squeeze(0)  # HxW float32 on CPU
            else:
                dmap = out_f.squeeze(0).squeeze(0)

            orig_h, orig_w = frame.shape[:2]
            
            # FIXED: Correct feed size extraction (width, height order)
            if FRAME_RESIZE is not None:
                feed_w, feed_h = FRAME_RESIZE  # (512, 384) -> width=512, height=384
            else:
                feed_h, feed_w = inp.shape[2], inp.shape[3]  # H, W from tensor

            # FIXED: Exact count calculation logic
            if UPSAMPLE_DMAP:
                dmap_unsq = dmap.unsqueeze(0).unsqueeze(0)
                dmap_up = F.interpolate(dmap_unsq, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
                dmap_up = dmap_up.squeeze(0).squeeze(0)
                count_val = float(dmap_up.sum().item())
            elif SCALE_BY_AREA:
                raw_count = float(dmap.sum().item())
                scale = (orig_h * orig_w) / (feed_h * feed_w)
                count_val = raw_count * scale
            else:
                count_val = float(dmap.sum().item())

            # smoothing
            if SMOOTH_WINDOW and SMOOTH_WINDOW > 1:
                smoother.append(count_val)
                if len(smoother) > SMOOTH_WINDOW:
                    smoother.pop(0)
                display_count = sum(smoother) / len(smoother)
            else:
                display_count = count_val

            t_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            time_str = ms_to_time_str(t_ms)

            # camera hinderance check
            is_hindered, reason = check_camera_hinder(frame)
            alert_field = reason if is_hindered else ""
            if is_hindered:
                display_count = 0.00

            writer.writerow([
                today_str,
                time_str,
                f"{display_count:.3f}",
                alert_field
            ])

            processed += 1

            if time.time() - last_flush > FLUSH_INTERVAL:
                buf.flush()
                last_flush = time.time()

            if processed % PRINT_EVERY == 0:
                elapsed = time.time() - start_wall
                fps_eff = processed / elapsed if elapsed > 0 else 0
                print(f"Processed {processed} frames (effective FPS: {fps_eff:.2f}), last count {display_count:.2f}")

    cap.release()
    csv_text = buf.getvalue()
    buf.close()

    with RUNS_LOCK:
        RUNS[run_id]["csv"] = csv_text
        RUNS[run_id]["done"] = True

    print("Finished. Stored CSV in memory for run_id:", run_id)

def run_processing(video_path: str, run_id: str):
    with RUNS_LOCK:
        RUNS[run_id] = {"csv": "", "done": False}
    process_video_to_csv(video_path, run_id)
    try:
        os.remove(video_path)
    except OSError:
        pass

# FastAPI app (unchanged)
app = FastAPI(title="Video Analytics API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/process_video/")
async def process_video(background_tasks: BackgroundTasks, video: UploadFile = File(...)):
    run_id = str(int(time.time() * 1000))
    print(f"[process_video] run_id={run_id}")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        temp_video_path = tmp.name
        print(f"[process_video] writing temp video to: {temp_video_path}")
        while True:
            chunk = await video.read(1024 * 1024)
            if not chunk:
                break
            tmp.write(chunk)

    background_tasks.add_task(run_processing, temp_video_path, run_id)
    return {"status": "processing_started", "run_id": run_id}

@app.get("/crowd_txt/{run_id}")
async def crowd_txt(run_id: str):
    with RUNS_LOCK:
        run_info = RUNS.get(run_id)
    if not run_info:
        return JSONResponse({"csv": "", "done": True})
    return JSONResponse({
        "csv": run_info.get("csv", ""),
        "done": run_info.get("done", False)
    })

