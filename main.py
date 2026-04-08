from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import cv2
import numpy as np
import tempfile
import shutil
import mediapipe as mp
import subprocess
import os
from kba_features import extract_features, interpret_biomechanics

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

mp_pose = mp.solutions.pose

MAX_TRACKING_SECONDS = 10.0
PLAYER_BOX_RATIO     = 0.18
OUT_W    = 720
OUT_H    = 1280
CLIP_PRE  = 1.5
CLIP_POST = 4.5

def draw_overlay(frame, player_name, frame_idx, total_frames):
    h, w = frame.shape[:2]
    orange = (26, 92, 255)
    black  = (0, 0, 0)
    white  = (245, 232, 204)
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), orange, 4)
    cv2.rectangle(frame, (0, 0), (w, 68), black, -1)
    cv2.rectangle(frame, (0, 0), (5, 68), orange, -1)
    cv2.putText(frame, 'HOOPVISION', (18, 44), cv2.FONT_HERSHEY_DUPLEX, 1.1, orange, 2, cv2.LINE_AA)
    cv2.putText(frame, 'HIGHLIGHTS', (w - 148, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.55, white, 1, cv2.LINE_AA)
    cv2.rectangle(frame, (0, h - 130), (w, h), black, -1)
    cv2.rectangle(frame, (0, h - 130), (5, h), orange, -1)
    name_upper = player_name.upper()
    cv2.putText(frame, name_upper, (18, h - 76), cv2.FONT_HERSHEY_DUPLEX, 1.3, white, 2, cv2.LINE_AA)
    cv2.putText(frame, 'HIGHLIGHT', (18, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, orange, 2, cv2.LINE_AA)
    progress = frame_idx / max(total_frames, 1)
    bar_w = int(w * progress)
    cv2.rectangle(frame, (0, h - 8), (w, h), (20, 20, 20), -1)
    cv2.rectangle(frame, (0, h - 8), (bar_w, h), orange, -1)
    return frame

def make_highlight(src_path, t_event, player_name, out_path):
    cap = cv2.VideoCapture(src_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    fw  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    start_f = int(max(0, t_event - CLIP_PRE) * fps)
    end_f   = int((t_event + CLIP_POST) * fps)
    total_f = end_f - start_f
    tmp_out = out_path.replace('.mp4', '_raw.mp4')
    fourcc  = cv2.VideoWriter_fourcc(*'mp4v')
    writer  = cv2.VideoWriter(tmp_out, fourcc, fps, (OUT_W, OUT_H))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret or (start_f + frame_idx) > end_f:
            break
        canvas = np.zeros((OUT_H, OUT_W, 3), dtype=np.uint8)
        scale  = OUT_W / fw
        new_h  = int(fh * scale)
        resized = cv2.resize(frame, (OUT_W, new_h), interpolation=cv2.INTER_AREA)
        if new_h >= OUT_H:
            crop_y = (new_h - OUT_H) // 2
            canvas = resized[crop_y:crop_y + OUT_H, :]
        else:
            y_off = (OUT_H - new_h) // 2
            canvas[y_off:y_off + new_h, :] = resized
        canvas = draw_overlay(canvas, player_name, frame_idx, total_f)
        writer.write(canvas)
        frame_idx += 1
    cap.release()
    writer.release()
    subprocess.run([
        'ffmpeg', '-y', '-i', tmp_out,
        '-vcodec', 'libx264', '-crf', '26',
        '-preset', 'fast',
        '-movflags', '+faststart',
        '-an',
        out_path
    ], capture_output=True)
    if os.path.exists(tmp_out):
        os.remove(tmp_out)
    return os.path.exists(out_path)

@app.get("/")
def home():
    return {"status": "ok"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/export")
async def export_highlight(
    file: UploadFile = File(...),
    t: float = Form(...),
    player_name: str = Form("Jugador"),
):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as src:
        shutil.copyfileobj(file.file, src)
        src_path = src.name
    out_path = src_path.replace('.mp4', '_highlight.mp4')
    ok = make_highlight(src_path, t, player_name, out_path)
    if os.path.exists(src_path):
        os.remove(src_path)
    if not ok:
        return {"error": "No se pudo generar el highlight"}
    return FileResponse(
        out_path,
        media_type="video/mp4",
        filename=f"{player_name.replace(' ', '_')}_highlight.mp4",
        background=None
    )

@app.post("/upload")
async def upload_video(
    file: UploadFile = File(...),
    player_x: float = Form(0.5),
    player_y: float = Form(0.5)
):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    cap = cv2.VideoCapture(tmp_path)
    fps      = cap.get(cv2.CAP_PROP_FPS) or 30
    f_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    f_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    box_w = int(f_width  * PLAYER_BOX_RATIO)
    box_h = int(f_height * PLAYER_BOX_RATIO * 2.2)
    bx    = max(0, int(player_x * f_width  - box_w / 2))
    by    = max(0, int(player_y * f_height - box_h / 2))
    init_bbox = (bx, by, box_w, box_h)

    ret, first_frame = cap.read()
    if not ret:
        cap.release()
        return {"total": 0, "events": [], "note": "No se pudo leer el video"}

    tracker = cv2.TrackerCSRT_create()
    tracker.init(first_frame, init_bbox)

    tracking_active = True
    frame_idx       = 1
    wrist_history   = []
    all_keypoints   = []

    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.4,
        min_tracking_confidence=0.4
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        t = frame_idx / fps
        if tracking_active:
            ok, bbox = tracker.update(frame)
            if t > MAX_TRACKING_SECONDS:
                tracking_active = False
            if not ok:
                frame_idx += 1
                all_keypoints.append([])
                continue
            tx, ty, tw, th = [int(v) for v in bbox]
        else:
            tx, ty, tw, th = 0, 0, f_width, f_height

        if frame_idx % 3 == 0:
            margin = int(tw * 0.3)
            x1 = max(0, tx - margin)
            y1 = max(0, ty - margin)
            x2 = min(f_width,  tx + tw + margin)
            y2 = min(f_height, ty + th + margin)
            roi = frame[y1:y2, x1:x2]
            rgb     = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)
            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                all_keypoints.append([
                    {"x": l.x, "y": l.y, "z": l.z, "visibility": l.visibility}
                    for l in lm
                ])
                vis_r = lm[12].visibility + lm[14].visibility + lm[16].visibility
                vis_l = lm[11].visibility + lm[13].visibility + lm[15].visibility
                if vis_r >= vis_l:
                    sh, el, wr = lm[12], lm[14], lm[16]
                else:
                    sh, el, wr = lm[11], lm[13], lm[15]
                wrist_history.append({
                    "t":   round(t, 3),
                    "wy":  wr.y,
                    "ey":  el.y,
                    "sy":  sh.y,
                    "wx":  wr.x,
                    "vis": max(vis_r, vis_l) / 3.0
                })
            else:
                all_keypoints.append([])
        frame_idx += 1

    cap.release()
    pose.close()

    kba_raw      = extract_features(all_keypoints, fps=fps / 3)
    kba_coaching = interpret_biomechanics(kba_raw)

    if len(wrist_history) < 5:
        return {"total": 0, "events": [], "note": "No se detectaron poses"}

    COOLDOWN = 7.0
    MIN_RISE = 0.12
    shots  = []
    last_t = -COOLDOWN
    n      = len(wrist_history)

    for i in range(2, n - 2):
        cur  = wrist_history[i]
        prev = wrist_history[i - 2]
        nxt  = wrist_history[i + 2]
        t   = cur["t"]
        wy  = cur["wy"]
        sy  = cur["sy"]
        vis = cur["vis"]
        if t - last_t < COOLDOWN:
            continue
        if vis < 0.4:
            continue
        wrist_above_shoulder = wy < sy - MIN_RISE
        is_peak = wy < prev["wy"] - 0.02 and wy < nxt["wy"] - 0.01
        if wrist_above_shoulder and is_peak:
            rise  = sy - wy
            ang   = round(35 + rise * 80, 1)
            ang   = max(20, min(70, ang))
            speed = round(prev["wy"] - wy, 3)
            shots.append({
                "t":     round(t, 2),
                "angle": ang,
                "speed": speed,
                "rise":  round(rise, 3),
                "vis":   round(vis, 2)
            })
            last_t = t

    events = []
    for s in shots:
        ang = s["angle"]
        if 42 <= ang <= 52:
            quality = "good"
        elif 38 <= ang <= 56:
            quality = "warn"
        else:
            quality = "bad"
        score = round(0.5 + (s["rise"] * 2), 3)
        score = max(0.1, min(1.0, score))
        events.append({
            "t":       s["t"],
            "score":   score,
            "angle":   ang,
            "quality": quality,
            "rise":    s["rise"],
            "source":  "mediapipe"
        })

    return {
        "total":        len(events),
        "events":       events,
        "source":       "mediapipe_pose+csrt",
        "biomechanics": kba_coaching,
    }
