# fast_count_to_csv.py
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as T
import numpy as np
import time
import csv
from threading import Thread
from queue import Queue
import torch.nn.functional as F

# ---------------- USER SETTINGS ----------------
MODEL_PATH = r"E:\BTP-1\MCNN\crowd_counting.pth"
VIDEO_IN   = r"E:\BTP-1\MCNN\people.mp4"
OUT_CSV    = r"E:\BTP-1\MCNN\count4.csv"

USE_GPU = True                      # use CUDA if available
USE_FP16_IF_CUDA = True             # use float16 if using CUDA (speeds inference)
FRAME_RESIZE = (512, 384)           # (width, height) fed to model; set None to use original size
SKIP_FRAMES = 10                     # process every (SKIP_FRAMES)th frame; 1 = every frame
SMOOTH_WINDOW = 3                   # moving average window for display (0 or 1 disables)
QUEUE_MAXSIZE = 6                   # frame queue size for reader thread
FLUSH_INTERVAL = 2.0                # seconds between CSV flushes
PRINT_EVERY = 50                    # print progress every N processed frames
UPSAMPLE_DMAP = False        # True = upsample density map to original feed size then sum (best)
SCALE_BY_AREA = True       # True = multiply sum by area ratio (cheaper)
# If UPSAMPLE_DMAP is True, SCALE_BY_AREA is ignored.
# ------------------------------------------------

device = torch.device("cuda" if (USE_GPU and torch.cuda.is_available()) else "cpu")
print("Device:", device)

# --- Your model class (same as training) ---
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

# --- model loading helper (robust) ---
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
    # fix module. prefix
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
    # convert model to fp16 where supported
    model.half()

# --- fast preprocessing: cv2 resize + convert to tensor (no PIL) ---
def frame_to_tensor(frame_bgr):
    # frame_bgr: HxWx3 uint8
    if FRAME_RESIZE is not None:
        frame_bgr = cv2.resize(frame_bgr, FRAME_RESIZE, interpolation=cv2.INTER_AREA)
    # BGR -> RGB
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    # convert to float and scale to [0,1], reorder to CxHxW
    arr = np.asarray(rgb, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2,0,1))
    tensor = torch.from_numpy(arr).unsqueeze(0)  # 1x3xHxW
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

# --- threaded frame reader to decouple IO from processing ---
class FrameReader(Thread):
    def __init__(self, path, q, max_wait=0.1):
        super().__init__(daemon=True)
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open video: " + path)
        self.q = q
        self.running = True
    def run(self):
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                # signal end
                self.q.put(None)
                break
            # push frame and the current video timestamp in ms
            t_ms = self.cap.get(cv2.CAP_PROP_POS_MSEC)
            self.q.put((frame, t_ms))
        self.cap.release()

# --- main loop: consume frames, run model, write CSV ---
q = Queue(maxsize=QUEUE_MAXSIZE)
reader = FrameReader(VIDEO_IN, q)
reader.start()

csv_file = open(OUT_CSV, 'w', newline='', buffering=1)
writer = csv.writer(csv_file)
writer.writerow(['timestamp_ms', 'frame_index', 'count'])  # header

smoother = []
processed = 0
last_flush = time.time()
start_wall = time.time()

with torch.no_grad():
    while True:
        item = q.get()
        if item is None:
            print("End of video reached.")
            break
        frame, t_ms = item
        # frame skipping
        if processed % SKIP_FRAMES != 0:
            processed += 1
            continue

        # prepare tensor
        inp = frame_to_tensor(frame)

        # forward
        out = model(inp)
        # ensure float32 on CPU for summing if needed
        if device.type == 'cuda' and USE_FP16_IF_CUDA:
            dmap = out.float().squeeze(0).squeeze(0)
        else:
            dmap = out.squeeze(0).squeeze(0)

        orig_h, orig_w = frame.shape[:2]
        # feed size is size used to build `inp` tensor
        if FRAME_RESIZE is not None:
            feed_w, feed_h = FRAME_RESIZE
        else:
            # if we padded to multiples of 4 earlier then use that size
            feed_h, feed_w = inp.shape[2], inp.shape[3]

        # get density map tensor on CPU as float32
        if device.type == 'cuda' and USE_FP16_IF_CUDA:
            dmap = out.float().squeeze(0).squeeze(0)   # Hd x Wd
        else:
            dmap = out.squeeze(0).squeeze(0)

        # Option A: upsample density map to original frame size (recommended)
        if UPSAMPLE_DMAP:
            # dmap shape: (Hd, Wd) -> add batch and channel dims for interpolate
            dmap_unsq = dmap.unsqueeze(0).unsqueeze(0)  # 1x1xHd xWd
            # target size should be (orig_h, orig_w)
            dmap_up = F.interpolate(dmap_unsq, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
            dmap_up = dmap_up.squeeze(0).squeeze(0)
            count_val = float(dmap_up.sum().item())

        # Option B: area scaling (faster, approximate)
        elif SCALE_BY_AREA:
            raw_count = float(dmap.sum().item())
            scale = (orig_h * orig_w) / (feed_h * feed_w)
            count_val = raw_count * scale

        # Option C: no correction (fastest, may be inaccurate)
        else:
            count_val = float(dmap.sum().item())
        # count_val = float(dmap.sum().item())

        # smoothing
        if SMOOTH_WINDOW and SMOOTH_WINDOW > 1:
            smoother.append(count_val)
            if len(smoother) > SMOOTH_WINDOW:
                smoother.pop(0)
            display_count = sum(smoother) / len(smoother)
        else:
            display_count = count_val

        # write to CSV: timestamp in ms, frame index, count
        time_str = ms_to_time_str(t_ms)
        writer.writerow([time_str, processed, f"{display_count:.3f}"])

        # writer.writerow([int(t_ms), processed, f"{display_count:.3f}"])

        processed += 1

        # periodic flush to disk
        if time.time() - last_flush > FLUSH_INTERVAL:
            csv_file.flush()
            last_flush = time.time()

        # basic console progress
        if processed % PRINT_EVERY == 0:
            elapsed = time.time() - start_wall
            fps_eff = processed / elapsed if elapsed > 0 else 0
            print(f"Processed {processed} frames (effective FPS: {fps_eff:.2f}), last count {display_count:.2f}")

# cleanup
csv_file.flush()
csv_file.close()
print("Finished. Wrote counts to:", OUT_CSV)