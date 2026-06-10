# =============================================================================
# data_prep/stage3_build_dataset.py
# -----------------------------------------------------------------------------
# STAGE 3 of the staged face harvester -- the one that actually cuts clips.
#
# Input: data_prep/segments_scored.json (ranked segments from stage 2).
#
# The CONSISTENCY FILTER is the whole point of this stage and what makes the
# trained avatar look SHARP rather than blurry:
#   1. Compute an InsightFace embedding for each segment's representative face.
#   2. Cluster segments by appearance (same look / lighting / camera).
#   3. Pick the LARGEST consistent cluster as the base identity.
#   4. From that cluster only, take the highest-quality segments until
#      TARGET_MINUTES is reached, then STOP.
#   5. Cut those segments from the FULL-RES source (re-downloaded per selected
#      video) with frame-accurate ffmpeg + audio, into face_segments/.
#
# Outlier-looking segments are rejected even if individually high quality:
# mixing looks is what blurs an avatar.
# =============================================================================

import os
import sys
import json
import shutil
import subprocess

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np
import cv2

try:
    import yt_dlp
except ImportError:
    print("[STAGE3] yt-dlp not installed. Run:  pip install yt-dlp")
    raise

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
DATA_PREP_DIR = os.path.dirname(os.path.abspath(__file__))
SEGMENTS_IN = os.path.join(DATA_PREP_DIR, "segments_scored.json")
SCAN_STATE_IN = os.path.join(DATA_PREP_DIR, "scan_state.json")
FULL_VIDEO_DIR = os.path.join(DATA_PREP_DIR, "full_videos")
OUT_DIR = os.path.join(DATA_PREP_DIR, "face_segments")
SELECTED_OUT = os.path.join(DATA_PREP_DIR, "selected_segments.json")

TARGET_MINUTES = 25.0          # enough for SyncTalk, not more -- then STOP
CLUSTER_DISTANCE = 0.55        # cosine distance threshold for "same look"
CUT_CRF = 18                   # near-lossless re-encode for frame-accurate cuts

DOWNLOAD_FORMAT = "bestvideo[ext=mp4]+bestaudio/best"


def ffmpeg_exe():
    """Resolve an ffmpeg binary: PATH first, else the imageio_ffmpeg bundle.

    Needed both for yt-dlp stream merging and for our own frame-accurate cut;
    ffmpeg is not on PATH on this box.
    """
    import shutil
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


# -----------------------------------------------------------------------------
# EMBEDDINGS (InsightFace)
# -----------------------------------------------------------------------------
def build_face_app():
    """Initialise an InsightFace FaceAnalysis app (buffalo_l, CPU-safe)."""
    try:
        from insightface.app import FaceAnalysis
    except ImportError:
        print("[STAGE3] insightface not installed. Run:  pip install insightface")
        raise
    app = FaceAnalysis(name="buffalo_l",
                       allowed_modules=["detection", "recognition"])
    # ctx_id=0 uses GPU if onnxruntime-gpu is present, else falls back to CPU.
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app


def embed_segment(app, rep_frame):
    """Return the normalized 512-d embedding for a segment's rep face, or None."""
    if not rep_frame or not os.path.exists(rep_frame):
        return None
    img = cv2.imread(rep_frame)
    if img is None:
        return None
    faces = app.get(img)
    if not faces:
        return None
    # Largest face wins (rep crops are already tight, but be safe).
    face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    emb = face.normed_embedding
    if emb is None:
        return None
    return np.asarray(emb, dtype=np.float32)


# -----------------------------------------------------------------------------
# CONSISTENCY CLUSTERING
# -----------------------------------------------------------------------------
def cluster_by_appearance(embeddings):
    """Cluster normalized embeddings by cosine distance.

    Returns an array of integer labels (one per embedding). Uses agglomerative
    clustering with a distance threshold so we don't have to guess the number
    of distinct looks ahead of time.
    """
    n = len(embeddings)
    if n == 0:
        return np.array([], dtype=int)
    if n == 1:
        return np.array([0], dtype=int)
    try:
        from sklearn.cluster import AgglomerativeClustering
    except ImportError:
        print("[STAGE3] scikit-learn not installed. Run:  pip install scikit-learn")
        raise

    X = np.vstack(embeddings).astype(np.float32)
    model = AgglomerativeClustering(
        n_clusters=None, distance_threshold=CLUSTER_DISTANCE,
        metric="cosine", linkage="average")
    return model.fit_predict(X)


def pick_consistent_cluster(labels, segments):
    """Return (chosen_label, indices) for the LARGEST cluster by total duration.

    We weight by duration, not count: the cluster that gives the most usable
    footage of one consistent look is the best training base.
    """
    by_label = {}
    for i, lab in enumerate(labels):
        by_label.setdefault(int(lab), []).append(i)
    best_label, best_dur = None, -1.0
    for lab, idxs in by_label.items():
        dur = sum(segments[i]["duration"] for i in idxs)
        if dur > best_dur:
            best_label, best_dur = lab, dur
    return best_label, by_label.get(best_label, []), by_label


# -----------------------------------------------------------------------------
# FULL-RES DOWNLOAD + FRAME-ACCURATE CUT
# -----------------------------------------------------------------------------
def download_full(video_id, url):
    """Download one video at full resolution into FULL_VIDEO_DIR (cached)."""
    existing = _find_full(video_id)
    if existing:
        return existing
    outtmpl = os.path.join(FULL_VIDEO_DIR, f"{video_id}.%(ext)s")
    ydl_opts = {
        "format": DOWNLOAD_FORMAT,
        "outtmpl": outtmpl,
        "merge_output_format": "mp4",
        "quiet": True,
        "no_warnings": True,
        "ignoreerrors": True,
        "noplaylist": True,
        "retries": 3,
        "fragment_retries": 3,
        "concurrent_fragment_downloads": 4,
    }
    ff = ffmpeg_exe()
    if ff:
        ydl_opts["ffmpeg_location"] = ff
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except Exception as exc:
        print(f"[STAGE3] full-res download failed for {video_id}: {exc}")
        return None
    return _find_full(video_id)


def _find_full(video_id):
    if not os.path.isdir(FULL_VIDEO_DIR):
        return None
    for f in os.listdir(FULL_VIDEO_DIR):
        name, ext = os.path.splitext(f)
        if name == video_id and ext.lower() in (".mp4", ".mkv", ".webm"):
            p = os.path.join(FULL_VIDEO_DIR, f)
            if os.path.getsize(p) > 0:
                return p
    return None


def cut_segment(src, start, end, out_path):
    """Frame-accurate cut [start, end] from src with audio, re-encoded.

    Output seeking (-ss after -i) keeps the cut frame-accurate; a near-lossless
    CRF re-encode avoids the keyframe snapping you get from stream-copy.
    Returns True on success.
    """
    duration = max(0.1, end - start)
    ff = ffmpeg_exe()
    if not ff:
        print("[STAGE3] ffmpeg not found (PATH or imageio_ffmpeg). Install ffmpeg.")
        return False
    cmd = [
        ff, "-y", "-hide_banner", "-loglevel", "error",
        "-i", src,
        "-ss", f"{start:.3f}", "-t", f"{duration:.3f}",
        "-c:v", "libx264", "-crf", str(CUT_CRF), "-preset", "veryfast",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "160k",
        "-avoid_negative_ts", "make_zero",
        out_path,
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError:
        print("[STAGE3] ffmpeg not found on PATH. Install ffmpeg.")
        return False
    if res.returncode != 0 or not os.path.exists(out_path):
        print(f"[STAGE3] ffmpeg cut failed ({start:.1f}-{end:.1f}): "
              f"{res.stderr.strip()[:160]}")
        return False
    return True


# -----------------------------------------------------------------------------
# DRIVER
# -----------------------------------------------------------------------------
def run():
    if not os.path.exists(SEGMENTS_IN):
        print(f"[STAGE3] {SEGMENTS_IN} not found. Run stage 2 first.")
        return []
    with open(SEGMENTS_IN, encoding="utf-8") as f:
        data = json.load(f)
    segments = data.get("segments", [])
    if not segments:
        print("[STAGE3] No segments to work with. Stage 2 found nothing usable.")
        return []

    for d in (FULL_VIDEO_DIR, OUT_DIR):
        os.makedirs(d, exist_ok=True)

    # --- 1. Embeddings ---------------------------------------------------
    print(f"[STAGE3] Embedding {len(segments)} segment faces (InsightFace)...")
    app = build_face_app()
    embeds, kept = [], []
    for seg in segments:
        emb = embed_segment(app, seg.get("rep_frame"))
        if emb is not None:
            embeds.append(emb)
            kept.append(seg)
    print(f"[STAGE3] {len(kept)}/{len(segments)} segments produced an embedding.")
    if not kept:
        print("[STAGE3] No embeddings -> cannot run consistency filter. Abort.")
        return []

    # --- 2 & 3. Cluster, choose the dominant consistent look -------------
    labels = cluster_by_appearance(embeds)
    chosen_label, chosen_idx, by_label = pick_consistent_cluster(labels, kept)
    chosen_minutes = sum(kept[i]["duration"] for i in chosen_idx) / 60.0
    print(f"[STAGE3] Found {len(by_label)} appearance cluster(s). "
          f"Largest = label {chosen_label} with {len(chosen_idx)} segment(s) / "
          f"{chosen_minutes:.1f} min. Other clusters rejected as inconsistent.")

    # --- 4. Take top-quality segments from the cluster until target ------
    cluster_segs = sorted((kept[i] for i in chosen_idx),
                          key=lambda s: s["score"], reverse=True)
    selected, sel_seconds = [], 0.0
    for seg in cluster_segs:
        if sel_seconds >= TARGET_MINUTES * 60.0:
            break
        selected.append(seg)
        sel_seconds += seg["duration"]
    print(f"[STAGE3] Selected {len(selected)} segment(s) = "
          f"{sel_seconds/60.0:.1f} min (target {TARGET_MINUTES:.0f}).")
    if sel_seconds < TARGET_MINUTES * 60.0:
        print(f"[STAGE3] NOTE: consistent cluster holds only "
              f"{sel_seconds/60.0:.1f} min (< {TARGET_MINUTES:.0f}). Using all "
              f"of it -- consider raising TAKE_TOP in stage 1 or scanning more.")

    # --- 5. Cut from full-res source ------------------------------------
    selected.sort(key=lambda s: (s["video_id"], s["start"]))
    written = []
    out_index = 0
    needed_videos = sorted({s["video_id"] for s in selected})
    print(f"[STAGE3] Cutting from {len(needed_videos)} source video(s) "
          f"at full resolution...")

    full_cache = {}
    for seg in selected:
        vid = seg["video_id"]
        if vid not in full_cache:
            full_cache[vid] = download_full(vid, seg["url"])
        src = full_cache[vid]
        if not src:
            print(f"[STAGE3] skipping segment from {vid} (no full-res source)")
            continue
        out_index += 1
        out_path = os.path.join(OUT_DIR, f"seg_{out_index:03d}.mp4")
        if cut_segment(src, seg["start"], seg["end"], out_path):
            rec = dict(seg)
            rec["out_file"] = out_path
            rec["cluster"] = int(chosen_label)
            written.append(rec)
            print(f"[STAGE3]   seg_{out_index:03d}.mp4  "
                  f"{seg['duration']:.1f}s  q={seg['quality']:.2f}  {vid}")

    # --- selection record (for stage 4) ----------------------------------
    scanned = _scanned_count()
    total_channel = _total_channel_count()
    payload = {
        "target_minutes": TARGET_MINUTES,
        "chosen_cluster": int(chosen_label),
        "n_clusters": len(by_label),
        "cluster_sizes": {str(k): len(v) for k, v in by_label.items()},
        "selected_minutes": round(sum(s["duration"] for s in written) / 60.0, 2),
        "videos_used": sorted({s["video_id"] for s in written}),
        "videos_scanned": scanned,
        "total_channel_videos": total_channel,
        "segments": written,
    }
    with open(SELECTED_OUT, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    final_min = payload["selected_minutes"]
    n_used = len(payload["videos_used"])
    skipped = (total_channel - scanned) if total_channel else "?"
    print(f"\n[STAGE3] Selected {len(written)} segments, {final_min:.1f} minutes, "
          f"from {n_used} video(s) (scanned {scanned}).")
    if total_channel:
        print(f"[STAGE3] Skipped {skipped} of {total_channel} channel videos "
              f"entirely -- never downloaded.")
    print(f"[STAGE3] Clips -> {OUT_DIR}")
    print(f"[STAGE3] Selection record -> {SELECTED_OUT}")
    return written


def _scanned_count():
    if os.path.exists(SCAN_STATE_IN):
        try:
            with open(SCAN_STATE_IN, encoding="utf-8") as f:
                return len(json.load(f).get("scanned_ids", []))
        except Exception:
            pass
    return 0


def _total_channel_count():
    if os.path.exists(SCAN_STATE_IN):
        try:
            with open(SCAN_STATE_IN, encoding="utf-8") as f:
                return json.load(f).get("total_channel_videos")
        except Exception:
            pass
    return None


if __name__ == "__main__":
    run()
