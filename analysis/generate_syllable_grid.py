"""
generate_syllable_grid.py
=========================
Generates a 6x10 video grid of syllable clips. Each cell shows random 
instances of a unique syllable, zoomed in 480x480px around the smoothed 
keypoint medoid, spanning -0.5s to +2.0s around onset. A green border 
is drawn while the target syllable is actively occurring.
"""

import argparse
import sys
import os
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import random
import keypoint_moseq as kpms

# ─────────────────────────────────────────────────────────────────────────────
# Data loaders
# ─────────────────────────────────────────────────────────────────────────────

def open_video_info(video_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        sys.exit(f"[ERROR] Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n_frames, fps


def load_keypoints_kpms(slp_path: str, project_dir: str, outlier_scale_factor: float = 6.0):
    coordinates, confidences, bodyparts = kpms.load_keypoints(slp_path, "sleap")
    if len(coordinates) == 0:
        sys.exit(f"[ERROR] No data in {slp_path}")
        
    key = list(coordinates.keys())[0]
    coords = coordinates[key]
    confs = confidences[key]
    node_names = list(bodyparts)

    coordinates_clean, _ = kpms.util.outlier_removal(
        coordinates={key: coords},
        confidences={key: confs},
        project_dir=project_dir,
        outlier_scale_factor=outlier_scale_factor,
        bodyparts=node_names,
        overwrite=True,
    )
    return coordinates_clean[key].astype(np.float32)


def load_syllables(csv_path: str, n_frames: int):
    df = pd.read_csv(csv_path, low_memory=False)
    df.columns = [c.strip().lower() for c in df.columns]
    syllables = np.full(n_frames, -1, dtype=int)

    if "frame" in df.columns:
        frames = df["frame"].values
        valid = (frames >= 0) & (frames < n_frames)
        if "syllable" in df.columns:
            syllables[frames[valid].astype(int)] = df.loc[valid, "syllable"].values
    else:
        length = min(n_frames, len(df))
        if "syllable" in df.columns:
            syllables[:length] = pd.to_numeric(df["syllable"][:length], errors="coerce").fillna(-1).astype(int)
            
    return syllables

# ─────────────────────────────────────────────────────────────────────────────
# Geometry and Array Helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_smoothed_medoids(keypoints, window=15):
    print("[INFO] Computing smoothed medoids for crop tracking...")
    n = len(keypoints)
    medoids = np.zeros((n, 2))
    
    for i in range(n):
        valid_pts = keypoints[i][~np.isnan(keypoints[i]).any(axis=1)]
        if len(valid_pts) > 0:
            diff = valid_pts[:, np.newaxis, :] - valid_pts[np.newaxis, :, :]
            dist = np.linalg.norm(diff, axis=-1).sum(axis=0)
            medoids[i] = valid_pts[np.argmin(dist)]
        else:
            medoids[i] = [np.nan, np.nan]
            
    df = pd.DataFrame(medoids).interpolate(method='linear', limit_direction='both')
    df.fillna(method='bfill', inplace=True)
    df.fillna(method='ffill', inplace=True)
    df.fillna(0, inplace=True)
    
    return df.rolling(window=window, center=True, min_periods=1).mean().values


def get_padded_crop(frame, cx, cy, size=480):
    h, w = frame.shape[:2]
    half = size // 2
    cx, cy = int(cx), int(cy)
    
    x1, y1 = cx - half, cy - half
    x2, y2 = cx + half, cy + half
    
    pad_l = max(0, -x1)
    pad_t = max(0, -y1)
    pad_r = max(0, x2 - w)
    pad_b = max(0, y2 - h)
    
    crop_x1 = max(0, x1)
    crop_y1 = max(0, y1)
    crop_x2 = min(w, x2)
    crop_y2 = min(h, y2)
    
    crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
    
    if pad_l > 0 or pad_t > 0 or pad_r > 0 or pad_b > 0:
        crop = cv2.copyMakeBorder(crop, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT, value=(0,0,0))
        
    return crop


def find_syllable_onsets(syllables):
    bouts = {i: [] for i in range(100)} 
    changes = np.where(syllables[:-1] != syllables[1:])[0] + 1
    starts = np.insert(changes, 0, 0)
    
    for st in starts:
        syl = syllables[st]
        if 0 <= syl < 100:
            bouts[syl].append(st)
            
    return bouts

# ─────────────────────────────────────────────────────────────────────────────
# Video Generation
# ─────────────────────────────────────────────────────────────────────────────

def render_grid(vid_file, keypoints, syllables, output_path, fps, 
                samples=20, duration=3.0, pre_onset=1.0, smooth_window=15):
    
    print("[INFO] Locating syllable onsets...")
    bouts = find_syllable_onsets(syllables)
    
    valid_syls = [s for s, st_frames in bouts.items() if len(st_frames) >= samples]
    valid_syls = sorted(valid_syls)[:60]
    
    if not valid_syls:
        sys.exit(f"[ERROR] No syllables found with at least {samples} instances.")
        
    sampled_bouts = {}
    for syl in valid_syls:
        sampled_bouts[syl] = random.sample(bouts[syl], samples)

    medoids = compute_smoothed_medoids(keypoints, window=smooth_window)

    pre_frames = int(pre_onset * fps)
    total_frames = int(duration * fps)
    
    cell_size = 480
    grid_cols, grid_rows = 10, 6
    out_w = grid_cols * cell_size
    out_h = grid_rows * cell_size
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))
    
    if not writer.isOpened():
        sys.exit(f"[ERROR] Failed to open VideoWriter for {output_path}")

    print(f"[INFO] Rendering grid ({out_w}x{out_h}) with {samples} instances per cell.")
    
    for inst_idx in range(samples):
        print(f"  Processing instance {inst_idx + 1}/{samples}...")
        caps = []
        
        for cell_idx in range(grid_rows * grid_cols):
            if cell_idx < len(valid_syls):
                syl_id = valid_syls[cell_idx]
                onset = sampled_bouts[syl_id][inst_idx]
                start_f = onset - pre_frames
                cap = cv2.VideoCapture(vid_file)
                cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, start_f))
                caps.append((cell_idx, syl_id, cap, start_f))
            else:
                caps.append((cell_idx, None, None, -1))

        for f in range(total_frames):
            canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
            
            for cell_idx, syl_id, cap, start_f in caps:
                if cap is None:
                    continue
                    
                target_f = start_f + f
                if target_f < 0 or target_f >= len(medoids):
                    continue
                    
                ok, frame = cap.read()
                if ok:
                    medoid = medoids[target_f]
                    
                    if not np.isnan(medoid).any():
                        crop = get_padded_crop(frame, medoid[0], medoid[1], cell_size)
                        row = cell_idx // grid_cols
                        col = cell_idx % grid_cols
                        y1 = row * cell_size
                        x1 = col * cell_size
                        
                        canvas[y1:y1+cell_size, x1:x1+cell_size] = crop
                        
                        # Add white background for the text
                        text = f"Syllable {syl_id}"
                        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                        cv2.rectangle(canvas, (x1 + 5, y1 + 5), (x1 + 15 + tw, y1 + 15 + th), (255, 255, 255), -1)
                        cv2.putText(canvas, text, (x1 + 10, y1 + 10 + th), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                        
                        time_rel = (f - pre_frames) / fps
                        color = (0, 255, 0) if time_rel >= 0 else (0, 0, 255)
                        cv2.putText(canvas, f"{time_rel:+.1f}s", (x1 + 10, y1 + cell_size - 20), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                                    
                        # Draw green border if the current frame's syllable matches this cell's target syllable
                        if syllables[target_f] == syl_id:
                            border_thickness = 8
                            cv2.rectangle(canvas, 
                                          (x1 + border_thickness//2, y1 + border_thickness//2), 
                                          (x1 + cell_size - border_thickness//2, y1 + cell_size - border_thickness//2), 
                                          (0, 255, 0), border_thickness)

            writer.write(canvas)
            
        for _, _, cap, _ in caps:
            if cap: cap.release()
            
    writer.release()
    print(f"[INFO] Output saved to {output_path}")

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Generate a 6x10 video grid of syllable clips.")
    p.add_argument("--subject",  required=True)
    p.add_argument("--session",  required=True)
    p.add_argument("--kpms-dir", default="/Users/ramzyalmulla/research/smearlab/kpms")
    p.add_argument("--data-dir", default="/Volumes/CrucialRZA/smearlab/clickbait-loco/thermister/")
    p.add_argument("--model-dir", default="/Volumes/CrucialRZA/smearlab/clickbait-loco/thermister/")
    p.add_argument("--model-folder", default="labels_v003_0.5scaling")
    p.add_argument("--outlier-scale-factor", type=float, default=6.0)
    p.add_argument("--samples", type=int, default=20)
    p.add_argument("--smooth-window", type=int, default=15, 
                   help="Rolling window size for medoid smoothing.")
    
    return p.parse_args()


def resolve_paths(args):
    subject, session = args.subject, args.session
    data_dir, kpms_dir = Path(args.data_dir), Path(args.kpms_dir)
    model_dir = Path(args.model_dir)

    def _glob_one(root, pattern, label):
        matches = list(root.glob(pattern))
        if not matches:
            sys.exit(f"[ERROR] No {label} file found matching: {root / pattern}")
        return str(matches[0])

    slp_file  = _glob_one(model_dir, f"{args.model_folder}/**/[0-9]*{subject}_{session}.slp", "SLEAP .slp")
    vid_file  = _glob_one(data_dir, f"**/[0-9]*{subject}_{session}.avi", "video .avi")
    kpms_file = _glob_one(kpms_dir, f"**/[0-9]*{subject}_{session}.csv", "kpms .csv")

    out_dir = kpms_dir / "syllable_grids"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = str(out_dir / f"{subject}_{session}_grid_6x10.mp4")

    return slp_file, vid_file, kpms_file, out_file


def main():
    args = parse_args()
    slp_file, vid_file, kpms_file, out_file = resolve_paths(args)

    if os.path.exists(out_file):
        os.remove(out_file)

    n_frames, fps = open_video_info(vid_file)
    keypoints = load_keypoints_kpms(slp_file, args.kpms_dir, args.outlier_scale_factor)
    syllables = load_syllables(kpms_file, n_frames)
    
    if len(keypoints) > n_frames:
        keypoints = keypoints[:n_frames]
    if len(syllables) > n_frames:
        syllables = syllables[:n_frames]

    render_grid(
        vid_file=vid_file, 
        keypoints=keypoints, 
        syllables=syllables, 
        output_path=out_file, 
        fps=fps, 
        samples=args.samples,
        duration=2.5,
        pre_onset=0.5,
        smooth_window=args.smooth_window
    )

if __name__ == "__main__":
    main()