"""
kba_features.py — Kinematic Biomechanics Analysis
Extracts joint angles and wrist velocity from a per-frame keypoint list,
then returns plain-language coaching cues.
"""

import math
from typing import List, Dict, Any


# ── Geometry helpers ──────────────────────────────────────────────────────────

def _angle_3pts(a, b, c) -> float:
    """
    Return the angle (degrees) at point B formed by A-B-C.
    Each point is a dict with 'x', 'y'.
    """
    ax, ay = a["x"] - b["x"], a["y"] - b["y"]
    cx, cy = c["x"] - b["x"], c["y"] - b["y"]
    dot   = ax * cx + ay * cy
    mag_a = math.hypot(ax, ay)
    mag_c = math.hypot(cx, cy)
    if mag_a == 0 or mag_c == 0:
        return 0.0
    cos_val = max(-1.0, min(1.0, dot / (mag_a * mag_c)))
    return math.degrees(math.acos(cos_val))


def _pick_side(lm: List[Dict]) -> tuple:
    """
    Return (shoulder, elbow, wrist, hip) landmark dicts for the more-visible side.
    MediaPipe indices:
      11=L-shoulder  12=R-shoulder
      13=L-elbow     14=R-elbow
      15=L-wrist     16=R-wrist
      23=L-hip       24=R-hip
      25=L-knee      26=R-knee
    """
    if len(lm) < 27:
        return None
    vis_r = lm[12]["visibility"] + lm[14]["visibility"] + lm[16]["visibility"]
    vis_l = lm[11]["visibility"] + lm[13]["visibility"] + lm[15]["visibility"]
    if vis_r >= vis_l:
        return lm[12], lm[14], lm[16], lm[24], lm[26]   # R: sh, el, wr, hip, knee
    return lm[11], lm[13], lm[15], lm[23], lm[25]        # L


# ── Feature extraction ────────────────────────────────────────────────────────

def extract_features(
    all_keypoints: List[List[Dict]],
    fps: float = 10.0
) -> Dict[str, Any]:
    """
    Parameters
    ----------
    all_keypoints : list of frames; each frame is either [] (no pose) or
                    a list of 33 landmark dicts {x, y, z, visibility}.
    fps           : effective frame rate of the keypoint stream.

    Returns
    -------
    dict with numeric metrics (angles, velocity, frame count).
    """
    elbow_angles  = []
    knee_angles   = []
    wrist_y_vals  = []

    for lm in all_keypoints:
        if not lm:
            continue
        pts = _pick_side(lm)
        if pts is None:
            continue
        sh, el, wr, hip, knee_pt = pts

        # Elbow angle: shoulder → elbow → wrist
        ea = _angle_3pts(sh, el, wr)
        if ea > 0:
            elbow_angles.append(ea)

        # Knee angle: hip → knee → ankle  (ankle = knee_pt index + 2)
        # We stored knee_pt (index 25 or 26); ankle = 27 or 28
        ankle_idx = 27 if lm[25] == knee_pt else 28
        if ankle_idx < len(lm):
            ka = _angle_3pts(hip, knee_pt, lm[ankle_idx])
            if ka > 0:
                knee_angles.append(ka)

        wrist_y_vals.append(wr["y"])

    # Wrist velocity (peak frame-to-frame change in normalised Y, scaled to 1/s)
    wrist_vels = []
    for i in range(1, len(wrist_y_vals)):
        dy = abs(wrist_y_vals[i] - wrist_y_vals[i - 1])
        wrist_vels.append(dy * fps)

    def _safe(lst, fn):
        return round(fn(lst), 3) if lst else 0.0

    return {
        "elbow_angle_mean":    _safe(elbow_angles, lambda x: sum(x) / len(x)),
        "elbow_angle_min":     _safe(elbow_angles, min),
        "elbow_angle_max":     _safe(elbow_angles, max),
        "knee_angle_mean":     _safe(knee_angles, lambda x: sum(x) / len(x)),
        "wrist_velocity_peak": _safe(wrist_vels, max),
        "frames_analysed":     len(elbow_angles),
    }


# ── Coaching interpretation ───────────────────────────────────────────────────

def interpret_biomechanics(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Attach plain-language coaching strings to the numeric KBA dict.
    Returns the same dict extended with *_coaching keys.
    """
    out = dict(raw)

    # Elbow coaching  (optimal 80–120°)
    ea = raw["elbow_angle_mean"]
    if ea == 0:
        out["elbow_coaching"] = "Elbow angle could not be measured."
    elif 80 <= ea <= 120:
        out["elbow_coaching"] = f"Elbow angle solid ({ea:.0f}°) — within optimal 80–120° range."
    elif ea < 80:
        out["elbow_coaching"] = f"Elbow too bent ({ea:.0f}°) — try a more open arm position."
    else:
        out["elbow_coaching"] = f"Elbow too extended ({ea:.0f}°) — bend arm more on the set position."

    # Knee coaching  (optimal 130–160°; shallower = more bend)
    ka = raw["knee_angle_mean"]
    if ka == 0:
        out["knee_coaching"] = "Knee angle could not be measured."
    elif ka < 130:
        out["knee_coaching"] = f"Deep knee bend ({ka:.0f}°) — good power generation."
    elif 130 <= ka <= 155:
        out["knee_coaching"] = f"Knee bend acceptable ({ka:.0f}°) — slight deeper bend may help."
    else:
        out["knee_coaching"] = f"Minimal knee bend ({ka:.0f}°) — try bending knees more for better drive."

    # Wrist velocity / snap coaching  (threshold ~0.05 normalised units/s)
    wv = raw["wrist_velocity_peak"]
    if wv == 0:
        out["wrist_coaching"] = "Wrist velocity could not be measured."
    elif wv >= 0.06:
        out["wrist_coaching"] = "Wrist snap detected — good follow-through acceleration."
    elif wv >= 0.03:
        out["wrist_coaching"] = f"Moderate wrist snap ({wv:.3f}) — focus on flicking the wrist at release."
    else:
        out["wrist_coaching"] = f"Low wrist velocity ({wv:.3f}) — practise exaggerated follow-through drills."

    return out
