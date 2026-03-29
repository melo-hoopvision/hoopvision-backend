from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import tempfile
import shutil

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"status": "ok"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/upload")
async def upload_video(file: UploadFile = File(...), player_x: float = 0.5, player_y: float = 0.5):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
    cap = cv2.VideoCapture(tmp_path)
    motion_frames = []
    prev_frame = None
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_frame is not None:
            diff = cv2.absdiff(gray, prev_frame)
            score = float(np.mean(diff)) / 255.0
            t = round(frame_idx / fps, 2)
            motion_frames.append({"t": t, "score": score})
        prev_frame = gray
        frame_idx += 1
    cap.release()
    threshold = 0.04
    events = [f for f in motion_frames if f["score"] > threshold]
    filtered = []
    last_t = -2
    for e in events:
        if e["t"] - last_t > 1.0:
            filtered.append(e)
            last_t = e["t"]
    return {"total": len(filtered), "events": filtered}
    
    with tempfile.NamedTemporaryFile(delete=False) as temp:
        shutil.copyfileobj(file.file, temp)
        temp_path = temp.name

    cap = cv2.VideoCapture(temp_path)

    motion_frames = 0
    prev_frame = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_frame is not None:
            diff = cv2.absdiff(prev_frame, gray)
            _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
            motion = np.sum(thresh)

            if motion > 500000:
                motion_frames += 1

        prev_frame = gray

    cap.release()

    return {
        "status": "processed",
        "motion_frames": motion_frames
    }
