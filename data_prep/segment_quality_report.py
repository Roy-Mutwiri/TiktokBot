# =============================================================================
# data_prep/segment_quality_report.py
# -----------------------------------------------------------------------------
# STAGE 4: read the selected dataset (data_prep/selected_segments.json) and
# print a go / no-go report before you spend hours training.
#
# Reports:
#   - Pose distribution (yaw bins) across the chosen segments
#   - Lighting (brightness) + sharpness distribution
#   - Consistency cluster preview (how dominant the chosen look is)
#   - A blunt GO / NO-GO verdict with the reasons behind it
#
# Read-only: it inspects what stage 3 produced and judges it. It does not cut,
# download, or modify anything.
# =============================================================================

import os
import sys
import json

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np

DATA_PREP_DIR = os.path.dirname(os.path.abspath(__file__))
SELECTED_IN = os.path.join(DATA_PREP_DIR, "selected_segments.json")
SEGMENTS_IN = os.path.join(DATA_PREP_DIR, "segments_scored.json")
OUT_DIR = os.path.join(DATA_PREP_DIR, "face_segments")

# Go / no-go thresholds.
MIN_MINUTES = 15.0             # below this, SyncTalk struggles
GOOD_MINUTES = 22.0            # comfortably enough
MIN_DOMINANT_FRAC = 0.70       # chosen cluster should hold >=70% of scanned segs
MIN_MEAN_SHARP = 60.0          # mean face sharpness floor
BRIGHT_OK = (60.0, 210.0)      # acceptable mean-brightness band
MAX_FRONTAL_FRAC = 0.95        # a little pose variety is healthy, not a wall of
MIN_FRONTAL_FRAC = 0.50        # frontal frames; want mostly-frontal though


def _bar(frac, width=30):
    n = int(round(max(0.0, min(1.0, frac)) * width))
    return "#" * n + "." * (width - n)


def load_selected():
    if not os.path.exists(SELECTED_IN):
        print(f"[REPORT] {SELECTED_IN} not found. Run stage 3 first.")
        return None
    with open(SELECTED_IN, encoding="utf-8") as f:
        return json.load(f)


def pose_distribution(segs):
    """Bin mean yaw into left / frontal / right buckets."""
    bins = {"left (< -15)": 0, "frontal (+/-15)": 0, "right (> 15)": 0}
    for s in segs:
        y = s.get("mean_yaw", 0.0)
        if y < -15:
            bins["left (< -15)"] += 1
        elif y > 15:
            bins["right (> 15)"] += 1
        else:
            bins["frontal (+/-15)"] += 1
    return bins


def report():
    data = load_selected()
    if not data:
        return False
    segs = data.get("segments", [])
    if not segs:
        print("[REPORT] selected_segments.json has no segments. NO-GO.")
        return False

    minutes = data.get("selected_minutes", sum(s["duration"] for s in segs) / 60.0)
    files_present = (len([f for f in os.listdir(OUT_DIR) if f.endswith(".mp4")])
                     if os.path.isdir(OUT_DIR) else 0)

    yaws = np.array([s.get("mean_yaw", 0.0) for s in segs], dtype=float)
    sharps = np.array([s.get("mean_sharp", 0.0) for s in segs], dtype=float)
    brights = np.array([s.get("mean_bright", 0.0) for s in segs], dtype=float)
    areas = np.array([s.get("mean_area", 0.0) for s in segs], dtype=float)
    quals = np.array([s.get("quality", 0.0) for s in segs], dtype=float)

    print("\n[REPORT] ============ DATASET QUALITY REPORT ============")
    print(f"[REPORT] Selected segments : {len(segs)}  "
          f"({files_present} clip file(s) on disk)")
    print(f"[REPORT] Total footage     : {minutes:.1f} min "
          f"(target {data.get('target_minutes', 25)})")
    print(f"[REPORT] Source videos used: {len(data.get('videos_used', []))}")
    scanned = data.get("videos_scanned")
    total = data.get("total_channel_videos")
    if scanned and total:
        print(f"[REPORT] Efficiency        : scanned {scanned} of {total} "
              f"channel videos ({100.0*scanned/total:.1f}%); skipped the rest.")

    # --- pose ---
    print("\n[REPORT] -- Pose distribution (mean yaw) --")
    bins = pose_distribution(segs)
    for k, v in bins.items():
        frac = v / len(segs)
        print(f"[REPORT]   {k:<18} {v:>3}  {_bar(frac)} {100*frac:4.0f}%")
    frontal_frac = bins["frontal (+/-15)"] / len(segs)

    # --- lighting / sharpness ---
    print("\n[REPORT] -- Lighting & sharpness --")
    print(f"[REPORT]   brightness  mean {brights.mean():6.1f}  "
          f"min {brights.min():6.1f}  max {brights.max():6.1f}")
    print(f"[REPORT]   sharpness   mean {sharps.mean():6.1f}  "
          f"min {sharps.min():6.1f}  max {sharps.max():6.1f}")
    print(f"[REPORT]   face area   mean {100*areas.mean():5.1f}% of frame  "
          f"(min {100*areas.min():.1f}%)")
    print(f"[REPORT]   quality     mean {quals.mean():.3f}  min {quals.min():.3f}")

    # --- consistency ---
    print("\n[REPORT] -- Consistency (appearance clusters) --")
    sizes = data.get("cluster_sizes", {})
    n_clusters = data.get("n_clusters", len(sizes) or 1)
    chosen = str(data.get("chosen_cluster"))
    total_scanned_segs = sum(sizes.values()) if sizes else len(segs)
    dominant = sizes.get(chosen, len(segs))
    dom_frac = dominant / total_scanned_segs if total_scanned_segs else 1.0
    print(f"[REPORT]   appearance clusters found : {n_clusters}")
    print(f"[REPORT]   chosen (dominant) cluster : {chosen} "
          f"holds {dominant}/{total_scanned_segs} scanned segs")
    print(f"[REPORT]   dominance  {_bar(dom_frac)} {100*dom_frac:4.0f}% "
          f"(want >= {100*MIN_DOMINANT_FRAC:.0f}%)")

    # --- verdict ---
    reasons = []
    if minutes < MIN_MINUTES:
        reasons.append(f"only {minutes:.1f} min (< {MIN_MINUTES:.0f})")
    if dom_frac < MIN_DOMINANT_FRAC:
        reasons.append(f"look not consistent enough "
                       f"({100*dom_frac:.0f}% < {100*MIN_DOMINANT_FRAC:.0f}%)")
    if sharps.mean() < MIN_MEAN_SHARP:
        reasons.append(f"soft footage (sharpness {sharps.mean():.0f} "
                       f"< {MIN_MEAN_SHARP:.0f})")
    if not (BRIGHT_OK[0] <= brights.mean() <= BRIGHT_OK[1]):
        reasons.append(f"lighting off (brightness {brights.mean():.0f} "
                       f"outside {BRIGHT_OK})")
    if frontal_frac < MIN_FRONTAL_FRAC:
        reasons.append(f"too few frontal segments ({100*frontal_frac:.0f}%)")
    if frontal_frac > MAX_FRONTAL_FRAC and len(segs) > 5:
        reasons.append("almost no pose variety (all dead-frontal)")

    print("\n[REPORT] ============ VERDICT ============")
    if not reasons:
        verdict = "GO" if minutes >= GOOD_MINUTES else "GO (marginal)"
        print(f"[REPORT] >>> {verdict} <<<  Dataset looks good for training.")
        print("[REPORT] Next: hand data_prep/face_segments/ to process_dataset.py")
        return True
    print("[REPORT] >>> NO-GO <<<  Issues found:")
    for r in reasons:
        print(f"[REPORT]   - {r}")
    print("[REPORT] Fix ideas: raise TAKE_TOP (stage 1) to scan more candidates, "
          "loosen/tighten CLUSTER_DISTANCE (stage 3), or relax frame gates "
          "(stage 2) if too few segments survived.")
    return False


if __name__ == "__main__":
    report()
