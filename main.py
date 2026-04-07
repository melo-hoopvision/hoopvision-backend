from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import tempfile
import shutil
import mediapipe as mp
from kba_features import extract_features, interpret_biomechanics

app = FastAPI()

app.add_middleware(
CORSMiddleware,
allow_origins=[”*”],
allow_credentials=True,
allow_methods=[”*”],
allow_headers=[”*”],
)

mp_pose = mp.solutions.pose

# ── Tracking config ───────────────────────────────────────────────────────────

MAX_TRACKING_SECONDS = 10.0   # Vía: evita drift acumulado
PLAYER_BOX_RATIO     = 0.18   # bounding box = 18% del ancho del frame

@app.get(”/”)
def home():
return {“status”: “ok”}

@app.get(”/health”)
def health():
return {“status”: “ok”}

@app.post(”/upload”)
async def upload_video(
file: UploadFile = File(…),
player_x: float = Form(0.5),   # centro X normalizado (0-1)
player_y: float = Form(0.5)    # centro Y normalizado (0-1)
):
# ── Guardar video temporal ────────────────────────────────────────────────
with tempfile.NamedTemporaryFile(delete=False, suffix=”.mp4”) as tmp:
shutil.copyfileobj(file.file, tmp)
tmp_path = tmp.name

```
cap = cv2.VideoCapture(tmp_path)
fps      = cap.get(cv2.CAP_PROP_FPS) or 30
f_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
f_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# ── Inicializar CSRT tracker en el primer frame ───────────────────────────
# Construimos bbox desde el centro x,y que manda el frontend
box_w = int(f_width  * PLAYER_BOX_RATIO)
box_h = int(f_height * PLAYER_BOX_RATIO * 2.2)   # más alto que ancho (cuerpo)
bx    = int(player_x * f_width  - box_w / 2)
by    = int(player_y * f_height - box_h / 2)
bx    = max(0, bx)
by    = max(0, by)
init_bbox = (bx, by, box_w, box_h)

ret, first_frame = cap.read()
if not ret:
    cap.release()
    return {"total": 0, "events": [], "note": "No se pudo leer el video"}

tracker = cv2.TrackerCSRT_create()
tracker.init(first_frame, init_bbox)

tracking_active = True
frame_idx       = 1    # ya leímos frame 0

wrist_history = []
all_keypoints = []

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

    # ── Actualizar tracker ────────────────────────────────────────────────
    if tracking_active:
        ok, bbox = tracker.update(frame)

        # Desactivar tracker después de MAX_TRACKING_SECONDS
        if t > MAX_TRACKING_SECONDS:
            tracking_active = False

        if not ok:
            # Tracker perdió al jugador — saltamos este frame
            frame_idx += 1
            all_keypoints.append([])
            continue

        tx, ty, tw, th = [int(v) for v in bbox]
    else:
        # Sin tracking: usamos frame completo (comportamiento original)
        tx, ty, tw, th = 0, 0, f_width, f_height

    # ── Procesar 1 de cada 3 frames ───────────────────────────────────────
    if frame_idx % 3 == 0:
        # Recortar ROI del jugador trackeado (con margen)
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

# ── KBA ───────────────────────────────────────────────────────────────────
kba_raw      = extract_features(all_keypoints, fps=fps / 3)
kba_coaching = interpret_biomechanics(kba_raw)

if len(wrist_history) < 5:
    return {"total": 0, "events": [], "note": "No se detectaron poses"}

# ── Detectar gestos de tiro ───────────────────────────────────────────────
COOLDOWN = 7.0
MIN_RISE = 0.12
MIN_DUR  = 0.25

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
        ey   = cur["ey"]
        rise = sy - wy
        ang  = round(35 + rise * 80, 1)
        ang  = max(20, min(70, ang))
        speed = round(prev["wy"] - wy, 3)

        shots.append({
            "t":     round(t, 2),
            "angle": ang,
            "speed": speed,
            "rise":  round(rise, 3),
            "vis":   round(vis, 2)
        })
        last_t = t

# ── Formatear respuesta ───────────────────────────────────────────────────
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
```
