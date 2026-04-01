from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import tempfile
import shutil
import mediapipe as mp

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── MediaPipe Pose ────────────────────────────────────────────────────────────
mp_pose = mp.solutions.pose

@app.get("/")
def home():
    return {"status": "ok"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/upload")
async def upload_video(
    file: UploadFile = File(...),
    player_x: float = Form(0.5),
    player_y: float = Form(0.5)
):
    # ── Guardar video temporal ────────────────────────────────────────────────
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    cap = cv2.VideoCapture(tmp_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_idx = 0

    # ── Historial de posiciones de muñeca ─────────────────────────────────────
    # Guardamos la posición Y normalizada de la muñeca derecha/izquierda
    wrist_history = []   # [{t, wy, ey, sy}]  wy=wrist, ey=elbow, sy=shoulder

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

        # Procesar 1 de cada 3 frames para velocidad (equiv ~10fps)
        if frame_idx % 3 == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)
            t = round(frame_idx / fps, 3)

            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                # Usamos hombro, codo y muñeca del lado con más visibilidad
                # Landmarks: 11=hombro_izq, 12=hombro_der
                #             13=codo_izq,  14=codo_der
                #             15=muñeca_izq,16=muñeca_der

                # Elegir el lado con mayor visibilidad
                vis_r = lm[12].visibility + lm[14].visibility + lm[16].visibility
                vis_l = lm[11].visibility + lm[13].visibility + lm[15].visibility

                if vis_r >= vis_l:
                    sh, el, wr = lm[12], lm[14], lm[16]
                else:
                    sh, el, wr = lm[11], lm[13], lm[15]

                wrist_history.append({
                    "t": t,
                    "wy": wr.y,   # Y normalizado (0=arriba, 1=abajo)
                    "ey": el.y,
                    "sy": sh.y,
                    "wx": wr.x,
                    "vis": max(vis_r, vis_l) / 3.0
                })

        frame_idx += 1

    cap.release()
    pose.close()

    if len(wrist_history) < 5:
        return {"total": 0, "events": [], "note": "No se detectaron poses"}

    # ── Detectar gestos de tiro ───────────────────────────────────────────────
    # Un tiro libre tiene esta firma:
    # 1. La muñeca SUBE (wy disminuye) por encima del hombro
    # 2. La muñeca llega a su punto más alto (peak)
    # 3. La muñeca BAJA después del release
    # Duración típica: 0.4s – 1.2s

    COOLDOWN   = 7.0   # segundos mínimos entre tiros
    MIN_RISE   = 0.12  # cuánto debe subir la muñeca sobre el hombro (Y units)
    MIN_DUR    = 0.25  # duración mínima del gesto en segundos

    shots = []
    last_t = -COOLDOWN
    n = len(wrist_history)

    for i in range(2, n - 2):
        cur  = wrist_history[i]
        prev = wrist_history[i - 2]
        nxt  = wrist_history[i + 2]

        t   = cur["t"]
        wy  = cur["wy"]
        sy  = cur["sy"]
        vis = cur["vis"]

        # Cooldown
        if t - last_t < COOLDOWN:
            continue

        # Visibilidad mínima
        if vis < 0.4:
            continue

        # La muñeca debe estar por ENCIMA del hombro (wy < sy en coords normalizadas)
        wrist_above_shoulder = wy < sy - MIN_RISE

        # La muñeca debe estar en un pico local (más arriba que sus vecinos)
        is_peak = wy < prev["wy"] - 0.02 and wy < nxt["wy"] - 0.01

        if wrist_above_shoulder and is_peak:
            # Calcular ángulo del codo
            ey = cur["ey"]
            # Ángulo aproximado basado en posición relativa
            rise = sy - wy          # qué tan arriba llegó la muñeca
            ang  = round(35 + rise * 80, 1)   # mapeo empírico → ~45° ideal
            ang  = max(20, min(70, ang))

            # Velocidad de subida (frames anteriores)
            speed = round(prev["wy"] - wy, 3)  # positivo = subiendo

            shots.append({
                "t":     round(t, 2),
                "angle": ang,
                "speed": speed,
                "rise":  round(rise, 3),
                "vis":   round(vis, 2)
            })
            last_t = t

    # ── Formatear respuesta para el frontend ──────────────────────────────────
    events = []
    for s in shots:
        ang = s["angle"]
        # Calidad del ángulo
        if 42 <= ang <= 52:
            quality = "good"
        elif 38 <= ang <= 56:
            quality = "warn"
        else:
            quality = "bad"

        # Score para compatibilidad con frontend existente
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
        "total":  len(events),
        "events": events,
        "source": "mediapipe_pose"
    }

