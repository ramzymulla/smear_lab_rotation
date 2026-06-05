"""
visualize_moseq.py
==================
Overlay SLEAP keypoints, keypoint-MoSeq syllable predictions, and a continuous
latent space UMAP on a mouse behavior video.
"""

import argparse
import sys
import warnings
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd
import keypoint_moseq as kpms

try:
    import umap
except ImportError:
    sys.exit("[ERROR] umap-learn is required for the latent visualizer. Run: pip install umap-learn")


# ─────────────────────────────────────────────────────────────────────────────
# Data loaders
# ─────────────────────────────────────────────────────────────────────────────

def open_video(video_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        sys.exit(f"[ERROR] Cannot open video: {video_path}")
    fps      = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    h        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    print(f"[INFO] Opened video: {n_frames} frames  {w}x{h}  {fps:.1f} fps  ({video_path})")
    return cap, n_frames, fps, h, w


def load_keypoints_kpms(slp_path: str, project_dir: str, outlier_scale_factor: float = 6.0):
    coordinates, confidences, bodyparts = kpms.load_keypoints(slp_path, "sleap")

    if len(coordinates) == 0:
        sys.exit(f"[ERROR] kpms.load_keypoints returned no data for {slp_path}")
    key    = list(coordinates.keys())[0]
    coords = coordinates[key]
    confs  = confidences[key]
    node_names = list(bodyparts)

    coordinates_clean, confidences_clean = kpms.util.outlier_removal(
        coordinates={key: coords},
        confidences={key: confs},
        project_dir=project_dir,
        outlier_scale_factor=outlier_scale_factor,
        bodyparts=node_names,
        overwrite=True,
    )
    coords = coordinates_clean[key].astype(np.float32)

    try:
        config = kpms.load_config(project_dir)
        skeleton_pairs = config.get("skeleton", [])
        name_to_idx = {n: i for i, n in enumerate(node_names)}
        edges = [
            (name_to_idx[a], name_to_idx[b])
            for a, b in skeleton_pairs
            if a in name_to_idx and b in name_to_idx
        ]
    except Exception as exc:
        warnings.warn(f"Could not load skeleton: {exc}. Falling back to linear chain.")
        edges = [(0, 1), (0, 2), (0, 3), (4, 5), (4, 6), (4, 7), (4, 8), (8, 0), (7, 9)]

    return coords, edges, node_names


def load_syllables_and_latents(csv_path: str, n_frames: int):
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]

    syllables = np.full(n_frames, -1, dtype=int)
    latent_cols = sorted([c for c in df.columns if c.startswith("latent_state")])
    latents = np.zeros((n_frames, len(latent_cols))) if latent_cols else None

    if "frame" in df.columns:
        frames = df["frame"].values
        valid = (frames >= 0) & (frames < n_frames)
        valid_frames = frames[valid].astype(int)
        
        if "syllable" in df.columns:
            syllables[valid_frames] = df.loc[valid, "syllable"].values
        if latents is not None:
            latents[valid_frames] = df.loc[valid, latent_cols].values
    else:
        length = min(n_frames, len(df))
        if "syllable" in df.columns:
            syllables[:length] = pd.to_numeric(df["syllable"][:length], errors="coerce").fillna(-1).astype(int)
        if latents is not None:
            latents[:length] = df[latent_cols].values[:length]

    print(f"[INFO] Loaded syllables & latents ({csv_path})")
    if latents is not None:
        print(f"[INFO] Found {len(latent_cols)} latent state dimensions.")
    else:
        print("[WARN] No 'latent_state' columns found. UMAP will be disabled.")
        
    return syllables, latents


# ─────────────────────────────────────────────────────────────────────────────
# Colour helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_syllable_cmap(syllable_ids):
    unique = sorted(set(s for s in syllable_ids if s >= 0))
    base = matplotlib.colormaps["tab20"].resampled(max(len(unique), 1))
    colors = {}
    for i, s in enumerate(unique):
        r, g, b, a = base(i % 20)
        colors[s] = (int(b * 255), int(g * 255), int(r * 255)) 
    return colors


# ─────────────────────────────────────────────────────────────────────────────
# Main render loop
# ─────────────────────────────────────────────────────────────────────────────

def render(cap, n_frames, h, w, keypoints, syllables, syllable_colors, edges, umap_coords, output_path, fps, window=300):
    timeline_h = 30
    zoom_h = 60
    margin = 15
    panel_h = timeline_h + zoom_h + (margin * 3)
    
    # Expand output width to fit a square UMAP visualizer if latents exist
    umap_size = h if umap_coords is not None else 0
    out_h = h + panel_h
    out_w = w + umap_size

    # Pre-render full timeline
    full_timeline_img = np.full((timeline_h, out_w, 3), 50, dtype=np.uint8)
    for fi, syl in enumerate(syllables):
        if syl >= 0 and syl in syllable_colors:
            x1 = int((fi / n_frames) * out_w)
            x2 = int(((fi + 1) / n_frames) * out_w)
            full_timeline_img[:, x1:max(x2, x1 + 1)] = syllable_colors[syl]

    # Pre-render static UMAP background
    pad = 20
    if umap_coords is not None:
        umap_bg = np.full((h, umap_size, 3), 20, dtype=np.uint8)
        for i, (ux, uy) in enumerate(umap_coords):
            s = int(syllables[i])
            if s >= 0 and s in syllable_colors:
                cx = int(ux * (umap_size - 2 * pad)) + pad
                cy = int(uy * (h - 2 * pad)) + pad
                cv2.circle(umap_bg, (cx, cy), 2, syllable_colors[s], -1)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    print(f"[INFO] Rendering {n_frames} frames via OpenCV → {output_path}")

    for fi in range(n_frames):
        if fi % 500 == 0:
            print(f"  {fi}/{n_frames}  ({fi / n_frames * 100:.0f}%)", flush=True)

        ok, frame = cap.read()
        if not ok:
            break

        canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        canvas[:h, :w] = frame

        syl = int(syllables[fi]) if fi < len(syllables) else -1
        active_color = syllable_colors.get(syl, (150, 150, 150))

        # Draw Skeleton & Keypoints with active syllable color
        if fi < len(keypoints):
            pts = keypoints[fi]
            for (a, b) in edges:
                if a < len(pts) and b < len(pts):
                    if np.isfinite(pts[a]).all() and np.isfinite(pts[b]).all():
                        cv2.line(canvas, 
                                 (int(pts[a][0]), int(pts[a][1])), 
                                 (int(pts[b][0]), int(pts[b][1])), 
                                 active_color, 2)

            for pt in pts:
                if np.isfinite(pt).all():
                    cv2.circle(canvas, (int(pt[0]), int(pt[1])), 4, active_color, -1)

        # Draw UMAP mapping
        if umap_coords is not None:
            canvas[:h, w:out_w] = umap_bg.copy()
            cv2.putText(canvas, "UMAP Latent Space", (w + 15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            
            if fi < len(umap_coords):
                ux, uy = umap_coords[fi]
                cx = w + pad + int(ux * (umap_size - 2 * pad))
                cy = pad + int(uy * (h - 2 * pad))
                cv2.circle(canvas, (cx, cy), 8, (255, 255, 255), -1)
                cv2.circle(canvas, (cx, cy), 5, active_color, -1)

        # Draw Syllable Text Overlay
        if syl >= 0:
            text = f"Syllable {syl}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            cv2.rectangle(canvas, (10, 10), (20 + tw, 20 + th), (0, 0, 0), -1)
            cv2.putText(canvas, text, (15, 15 + th), cv2.FONT_HERSHEY_SIMPLEX, 1, active_color, 2)

        # Draw Full Timeline
        y1_full = h + margin
        y2_full = y1_full + timeline_h
        canvas[y1_full:y2_full, :] = full_timeline_img
        
        cx_full = int((fi / n_frames) * out_w)
        cv2.line(canvas, (cx_full, y1_full), (cx_full, y2_full), (255, 255, 255), 2)

        # Draw Zoomed Timeline
        y1_zoom = y2_full + margin
        y2_zoom = y1_zoom + zoom_h
        cv2.rectangle(canvas, (0, y1_zoom), (out_w, y2_zoom), (30, 30, 30), -1)

        half = window // 2
        t0 = max(0, fi - half)
        t1 = min(n_frames, fi + half)
        window_len = window

        if t1 > t0:
            win_syls = syllables[t0:t1]
            for i, s in enumerate(win_syls):
                if s >= 0 and s in syllable_colors:
                    x1 = int((i / window_len) * out_w)
                    x2 = int(((i + 1) / window_len) * out_w)
                    cv2.rectangle(canvas, (x1, y1_zoom), (x2, y2_zoom), syllable_colors[s], -1)

        cx_zoom = int(((fi - t0) / window_len) * out_w)
        cv2.line(canvas, (cx_zoom, y1_zoom), (cx_zoom, y2_zoom), (255, 255, 255), 2)

        writer.write(canvas)

    writer.release()
    cap.release()
    print(f"[INFO] Done → {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Overlay SLEAP keypoints, MoSeq syllables, and Latent UMAP on a mouse video.")
    p.add_argument("--subject",  required=True)
    p.add_argument("--session",  required=True)
    p.add_argument("--kpms-dir", default="/Users/ramzyalmulla/research/smearlab/kpms")
    p.add_argument("--data-dir", default="/Volumes/CrucialRZA/smearlab/clickbait-loco/thermister/")
    p.add_argument("--model-dir", default="/Volumes/CrucialRZA/smearlab/clickbait-loco/thermister/")
    p.add_argument("--model-folder", default="labels_v003_0.5scaling")
    p.add_argument("--fps", type=float, default=None)
    p.add_argument("--window", type=int, default=300)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--outlier-scale-factor", type=float, default=6.0)
    return p.parse_args()


def resolve_paths(args):
    subject, session = args.subject, args.session
    data_dir, kpms_dir = Path(args.data_dir), Path(args.kpms_dir)
    model_dir, model = Path(args.model_dir), args.model_folder

    def _glob_one(root, pattern, label):
        matches = list(root.glob(pattern))
        if not matches:
            sys.exit(f"[ERROR] No {label} file found matching: {root / pattern}")
        return str(matches[0])

    slp_file  = _glob_one(model_dir, f"{model}/**/[0-9]*{subject}_{session}.slp", "SLEAP .slp")
    vid_file  = _glob_one(data_dir, f"**/[0-9]*{subject}_{session}.avi", "video .avi")
    kpms_file = _glob_one(kpms_dir, f"**/[0-9]*{subject}_{session}.csv", "kpms .csv")

    out_dir = kpms_dir / "annotated_videos"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = str(out_dir / f"{subject}_{session}_annotated.mp4")

    return slp_file, vid_file, kpms_file, out_file


def main():
    args = parse_args()
    slp_file, vid_file, kpms_file, out_file = resolve_paths(args)

    cap, n_frames, vid_fps, h, w = open_video(vid_file)
    fps = args.fps or vid_fps

    if args.max_frames:
        n_frames = min(n_frames, args.max_frames)

    keypoints, edges, node_names = load_keypoints_kpms(
        slp_file, project_dir=args.kpms_dir,
        outlier_scale_factor=args.outlier_scale_factor)
        
    syllables, latents = load_syllables_and_latents(kpms_file, n_frames)

    if len(keypoints) > n_frames:
        keypoints = keypoints[:n_frames]

    syllable_colors = build_syllable_cmap(syllables)
    
    umap_coords = None
    if latents is not None:
        print("[INFO] Computing UMAP on latent states... This may take a moment.")
        reducer = umap.UMAP(n_components=2, random_state=42)
        u_emb = reducer.fit_transform(latents)
        
        # Normalize between 0 and 1 for rendering boundaries
        u_min, u_max = u_emb.min(axis=0), u_emb.max(axis=0)
        umap_coords = (u_emb - u_min) / (u_max - u_min + 1e-8)

    render(cap, n_frames, h, w, keypoints, syllables, syllable_colors,
           edges, umap_coords, out_file, fps, window=args.window)

if __name__ == "__main__":
    main()