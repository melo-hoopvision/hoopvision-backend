“””
main.py — HoopVision Backend v2.0
FastAPI · FFmpeg Spider-Verse · Avatar Generator · Beat Engine
“””

import os
import json
import uuid
import tempfile
import subprocess
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response

from beat_generator import generate_beat_for_video

BASE_DIR   = Path(**file**).parent
PROFILES_PATH = BASE_DIR / “profiles.json”
TEMP_DIR   = Path(tempfile.gettempdir()) / “hoopvision”
TEMP_DIR.mkdir(parents=True, exist_ok=True)
HF_TOKEN   = os.environ.get(“HF_TOKEN”, “”)

app = FastAPI(title=“HoopVision API”, version=“2.0.0”)
app.add_middleware(CORSMiddleware, allow_origins=[”*”], allow_methods=[”*”], allow_headers=[”*”])

# ── UTILS ──────────────────────────────────────

def load_profiles():
with open(PROFILES_PATH) as f: data = json.load(f)
return {p[“profile_id”]: p for p in data[“profiles”]}

def get_video_duration(path):
r = subprocess.run([“ffprobe”,”-v”,“error”,”-show_entries”,“format=duration”,”-of”,“json”,path], capture_output=True, text=True)
return float(json.loads(r.stdout)[“format”][“duration”])

def ffmpeg(*args):
cmd = [“ffmpeg”,”-y”] + list(args)
r = subprocess.run(cmd, capture_output=True, text=True)
if r.returncode != 0: raise RuntimeError(r.stderr[-600:])
return r

# ── SPIDER-VERSE FILTER ────────────────────────

def spiderverse_filter(inp, out, profile):
r,g,b = profile[“color_rgb”]
sat = 2.1
ffmpeg(”-i”, inp,
“-vf”,
f”eq=saturation={sat}:contrast=1.45:brightness=0.04,”
f”colorchannelmixer=”
f”rr={1+(r/255-.5)*.3:.3f}:gg={1+(g/255-.5)*.3:.3f}:bb={1+(b/255-.5)*.3:.3f},”
“smartblur=lr=1.2:ls=1.0:lt=-28,”
“edgedetect=low=0.04:high=0.13:mode=wires,”
f”eq=saturation={sat}:contrast=1.2,”
“vignette=PI/4.5”,
“-c:v”,“libx264”,”-preset”,“fast”,”-crf”,“19”,”-an”, out)

def slow_mo(inp, out, factor):
ffmpeg(”-i”, inp, “-vf”, f”setpts={1/factor:.4f}*PTS”, “-an”, “-c:v”,“libx264”,”-preset”,“fast”, out)

def mix_beat(video, beat_wav, out):
ffmpeg(”-i”,video,”-i”,beat_wav,
“-filter_complex”,”[1:a]volume=0.85[b]”,
“-map”,“0:v”,”-map”,”[b]”,”-c:v”,“copy”,”-c:a”,“aac”,”-shortest”, out)

def add_watermark(inp, out, player_name, profile):
name = player_name.replace(”’”,”\’”).replace(”:”,r”:”)
pname = profile[“name”].replace(”’”,”\’”)
col = profile[“color”][1:]
ffmpeg(”-i”, inp, “-vf”,
f”drawtext=text=’{name}’:fontcolor=white:fontsize=40:x=w-tw-14:y=14:shadowcolor=black:shadowx=3:shadowy=3,”
f”drawtext=text=’{pname}’:fontcolor=0x{col}:fontsize=19:x=14:y=h-th-38:shadowcolor=black:shadowx=2:shadowy=2,”
f”drawtext=text=‘HOOPVISION’:fontcolor=white@0.45:fontsize=13:x=14:y=h-th-14:shadowcolor=black:shadowx=1:shadowy=1”,
“-c:v”,“libx264”,”-preset”,“fast”,”-c:a”,“copy”, out)

def trim_shot(inp, out, ts, before=2.0, after=3.0):
ffmpeg(”-i”,inp,”-ss”,str(max(0,ts-before)),”-t”,str(before+after),”-c:v”,“libx264”,”-preset”,“fast”,”-c:a”,“aac”, out)

# ── ENDPOINTS ──────────────────────────────────

@app.get(”/”)
def root(): return {“status”:“🕷 HoopVision v2.0”,“spider_verse”:True}

@app.get(”/health”)
def health():
try:
ffmpeg_ok = subprocess.run([“ffmpeg”,”-version”],capture_output=True).returncode == 0
except: ffmpeg_ok = False
return {“status”:“ok”,“ffmpeg”:ffmpeg_ok,“hf_token”:bool(HF_TOKEN),“profiles”:len(load_profiles())}

@app.get(”/profiles”)
def profiles(): return {“profiles”:list(load_profiles().values())}

@app.post(”/avatar”)
async def avatar_endpoint(
photo: UploadFile = File(…),
profile_id: str = Form(…),
player_name: str = Form(default=“PLAYER”),
):
from avatar_generator import generate_avatar_from_photo
profiles = load_profiles()
if profile_id not in profiles: raise HTTPException(404, “Perfil no encontrado”)
photo_bytes = await photo.read()
try:
avatar_bytes = generate_avatar_from_photo(photo_bytes, profile_id, player_name)
return Response(content=avatar_bytes, media_type=“image/png”,
headers={“X-Profile”:profile_id})
except Exception as e:
raise HTTPException(500, f”Avatar error: {e}”)

@app.post(”/export”)
async def export_highlight(
video: UploadFile = File(…),
profile_id: str = Form(…),
player_name: str = Form(default=“PLAYER”),
best_shot_timestamp: Optional[float] = Form(default=None),
trim_clip: bool = Form(default=False),
spider_verse: bool = Form(default=True),
):
profiles = load_profiles()
if profile_id not in profiles: raise HTTPException(404, “Perfil no encontrado”)
profile = profiles[profile_id]
jid = str(uuid.uuid4())[:8]
jdir = TEMP_DIR / jid
jdir.mkdir(parents=True, exist_ok=True)

```
try:
    ext = Path(video.filename).suffix or ".mp4"
    orig = str(jdir / f"orig{ext}")
    with open(orig,"wb") as f: f.write(await video.read())
    # ── 0. Convertir a H264 compatible (fix HEVC de iPhone) ──
    converted = str(jdir / "converted.mp4")
    try:
        ffmpeg("-i", orig,
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-c:a", "aac", "-movflags", "+faststart",
            converted)
        cur = converted
    except:
        cur = orig  # si falla, intentar con original

    if trim_clip and best_shot_timestamp is not None:
        t = str(jdir/"trimmed.mp4")
        trim_shot(cur, t, best_shot_timestamp)
        cur = t

    sl = str(jdir/"slowed.mp4")
    slow_mo(cur, sl, profile["slow_mo_factor"])
    cur = sl

    if spider_verse:
        sv = str(jdir/"sv.mp4")
        spiderverse_filter(cur, sv, profile)
        cur = sv

    dur = get_video_duration(cur)
    beat = generate_beat_for_video(profile["beat_style"], dur)
    bp = str(jdir/"beat.wav")
    with open(bp,"wb") as f: f.write(beat)

    mx = str(jdir/"mixed.mp4")
    mix_beat(cur, bp, mx)
    cur = mx

    final = str(jdir/"final.mp4")
    add_watermark(cur, final, player_name, profile)

    safe = player_name.lower().replace(" ","_")
    return FileResponse(final, media_type="video/mp4",
                        filename=f"hoopvision_{safe}_{jid}.mp4",
                        headers={"X-Profile":profile_id,"X-Spider-Verse":str(spider_verse)})

except RuntimeError as e: raise HTTPException(500, str(e))
except Exception as e:    raise HTTPException(500, f"Error: {e}")
```

@app.post(”/beat/preview”)
async def beat_preview(style: str = Form(…), duration: float = Form(default=8.0)):
if style not in [“trap_fast”,“hiphop_hard”,“edm_drop”]:
raise HTTPException(400,“Estilo inválido”)
beat = generate_beat_for_video(style, min(duration,30))
jid = str(uuid.uuid4())[:8]
p = str(TEMP_DIR/f”beat_{style}*{jid}.wav”)
with open(p,“wb”) as f: f.write(beat)
return FileResponse(p, media_type=“audio/wav”, filename=f”hoop*{style}.wav”)
