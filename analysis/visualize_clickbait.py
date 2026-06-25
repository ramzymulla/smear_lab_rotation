"""
visualize_clickbait.py
======================
Overlay SLEAP keypoints, keypoint-MoSeq syllable predictions, a continuous
latent space UMAP, and behavioral task structure on a mouse behavior video.
"""

import argparse
import sys
import warnings
from pathlib import Path
import subprocess
import os
import cv2
import matplotlib
import numpy as np
import pandas as pd
import math
import wave
import keypoint_moseq as kpms

try:
    import umap
except ImportError:
    pass  # Handled below if UMAP is requested


# ─────────────────────────────────────────────────────────────────────────────
# Data loaders & Generators
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
    df = pd.read_csv(csv_path, low_memory=False)
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
        print("[WARN] No 'latent_state' columns found.")
        
    return syllables, latents


def safe_parse_datetime(series):
    series = series.astype(str).str.strip()
    
    dt = pd.to_datetime(series, errors='coerce')
    if dt.notna().any():
        return dt
        
    try:
        dt = pd.to_datetime(series, format='mixed', errors='coerce')
    except ValueError:
        pass
        
    if dt.notna().any():
        return dt
        
    num = pd.to_numeric(series, errors='coerce')
    if num.notna().any():
        if num.max() > 1e11:
            return pd.to_datetime(num, unit='ms', errors='coerce')
        else:
            return pd.to_datetime(num, unit='s', errors='coerce')
            
    return dt


def load_task_data(eventsA_path, eventsB_path, kinematicsA_path, vid_ts_path, max_len):
    df_A = pd.read_csv(eventsA_path, low_memory=False)
    df_B = pd.read_csv(eventsB_path, low_memory=False)
    df_K = pd.read_csv(kinematicsA_path, low_memory=False)
    df_V_raw = pd.read_csv(vid_ts_path, header=None, low_memory=False)

    best_dt_v = None
    for col in df_V_raw.columns:
        dt = safe_parse_datetime(df_V_raw[col])
        if dt.notna().any():
            best_dt_v = dt
            break
            
    if best_dt_v is None:
        sys.exit(f"[ERROR] Could not parse any valid timestamps from {vid_ts_path}")

    df_K.columns = df_K.columns.str.strip()
    ts_col_K = 'timestamp' if 'timestamp' in df_K.columns else df_K.columns[0]
    dt_k = safe_parse_datetime(df_K[ts_col_K])
    valid_k = dt_k.dropna()
    if len(valid_k) == 0:
        sys.exit("[ERROR] Could not parse kinematics timestamps.")
    first_k_ts = valid_k.iloc[0]

    diffs = (best_dt_v - first_k_ts).abs()
    
    start_frame_idx = diffs.idxmin()
    if pd.isna(start_frame_idx):
        start_frame_idx = 0
    start_frame_idx = int(start_frame_idx)
    print(f"[INFO] First task timestamp matched to video frame {start_frame_idx}.")

    task_len = min(len(df_A), len(df_B), len(df_K))
    print(f"[INFO] Raw task length: {task_len} rows. Mapping 1:1 forward from video frame {start_frame_idx}.")

    df_A = df_A.iloc[:task_len]
    df_B = df_B.iloc[:task_len]
    df_K = df_K.iloc[:task_len]

    active_targets_raw = pd.to_numeric(df_A['activeTarget'].astype(str).replace(['None', 'nan', 'NaN'], np.nan), errors='coerce').fillna(-1).astype(int).values
    target_radii_raw = pd.to_numeric(df_A['targetRadius'], errors='coerce').fillna(0).values

    targets_fixed = np.copy(active_targets_raw)
    radii_fixed = np.copy(target_radii_raw)
    for i in range(1, task_len):
        if active_targets_raw[i] == -1 and active_targets_raw[i-1] != -1:
            targets_fixed[i] = active_targets_raw[i-1]
            radii_fixed[i] = target_radii_raw[i-1]
    
    active_targets_raw = targets_fixed
    target_radii_raw = radii_fixed

    reward_states = df_B['rewardState'].fillna(False)
    if reward_states.dtype == object:
        reward_states = reward_states.astype(str).str.lower() == 'true'
    else:
        reward_states = reward_states.astype(bool).values

    cx_raw = pd.to_numeric(df_K['centroid.X'], errors='coerce').fillna(np.nan).values
    cy_raw = pd.to_numeric(df_K['centroid.Y'], errors='coerce').fillna(np.nan).values

    states_raw = np.full(task_len, 2, dtype=int)
    for i in range(task_len):
        if active_targets_raw[i] != -1:
            states_raw[i] = 0
        elif reward_states[i]:
            states_raw[i] = 1

    final_n_frames = min(max_len, start_frame_idx + task_len)

    active_targets = np.full(final_n_frames, -1, dtype=int)
    target_radii = np.zeros(final_n_frames)
    cx = np.full(final_n_frames, np.nan)
    cy = np.full(final_n_frames, np.nan)
    states = np.full(final_n_frames, -1, dtype=int)
    
    fill_len = final_n_frames - start_frame_idx
    if fill_len > 0:
        active_targets[start_frame_idx : final_n_frames] = active_targets_raw[:fill_len]
        target_radii[start_frame_idx : final_n_frames] = target_radii_raw[:fill_len]
        cx[start_frame_idx : final_n_frames] = cx_raw[:fill_len]
        cy[start_frame_idx : final_n_frames] = cy_raw[:fill_len]
        states[start_frame_idx : final_n_frames] = states_raw[:fill_len]

    return {
        'active_targets': active_targets,
        'target_radii': target_radii,
        'cx': cx,
        'cy': cy,
        'states': states
    }, final_n_frames


def generate_grid_cells(w, h, n_tiles=8, scale=0.5):
    center_x = w / 2.0
    center_y = h / 2.0
    
    w_ratio = w / (math.sqrt(3) * (2 * n_tiles + 1))
    h_ratio = h / (1.5 * 2 * n_tiles + 2)
    size = min(w_ratio, h_ratio) * scale
    viz_radius = int(size * 0.45)
    
    cells = []
    for q in range(-n_tiles, n_tiles + 1):
        for r in range(-n_tiles, n_tiles + 1):
            if abs(q + r) <= n_tiles:
                px = center_x + size * math.sqrt(3) * (q + r / 2.0)
                py = center_y + size * 1.5 * r
                cells.append((int(px), int(py)))
    return cells, viz_radius


# ─────────────────────────────────────────────────────────────────────────────
# Audio Generation Helpers
# ─────────────────────────────────────────────────────────────────────────────

def generate_reward_beeps(states, fps, n_frames, output_wav):
    sample_rate = 44100
    duration = n_frames / fps
    audio = np.zeros(int(duration * sample_rate), dtype=np.float32)
    
    beep_duration = 0.15
    beep_freq = 800.0
    beep_samples = int(beep_duration * sample_rate)
    t_arr = np.linspace(0, beep_duration, beep_samples, endpoint=False)
    sine_wave = np.sin(2 * np.pi * beep_freq * t_arr)
    
    for i in range(1, len(states)):
        if states[i] == 1 and states[i-1] != 1:
            t_start = i / fps
            start_idx = int(t_start * sample_rate)
            end_idx = start_idx + beep_samples
            
            if end_idx <= len(audio):
                audio[start_idx:end_idx] += sine_wave
            else:
                audio[start_idx:] += sine_wave[:len(audio)-start_idx]
                
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val
        
    audio_int16 = np.int16(audio * 32767)
    
    with wave.open(output_wav, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_int16.tobytes())


# ─────────────────────────────────────────────────────────────────────────────
# Colour helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_syllable_cmap(syllable_ids):
    mpl = matplotlib
    unique = sorted(set(s for s in syllable_ids if s >= 0))
    colorStack = np.vstack([mpl.colormaps['tab20'].colors,
                            mpl.colormaps['tab20b'].colors,
                            mpl.colormaps['Pastel1'].colors,
                            mpl.colormaps['Pastel2'].colors,
                            mpl.colormaps['Set2'].colors,
                            mpl.colormaps['Set3'].colors])
    
    colors = {}
    for i, s in enumerate(unique):
        r, g, b = colorStack[i % len(colorStack)]
        colors[s] = (int(b * 255), int(g * 255), int(r * 255)) 
    return colors


def build_task_cmap():
    mpl = matplotlib
    colorStack = np.vstack([mpl.colormaps['tab10'].colors,
                            mpl.colormaps['tab20b'].colors,
                            mpl.colormaps['Pastel1'].colors,
                            mpl.colormaps['Pastel2'].colors])
    colors = {}

    for i, s in enumerate([0,1,2]):
        r, g, b = colorStack[i]
        colors[s] = (int(b * 255), int(g * 255), int(r * 255)) 
    return colors


# ─────────────────────────────────────────────────────────────────────────────
# Main render loop
# ─────────────────────────────────────────────────────────────────────────────

def render(cap, n_frames, h, w, keypoints, syllables, syllable_colors, edges, 
           umap_coords, task_data, task_colors, output_path, fps, window=300, 
           codec="mp4v", scale=1.0, color_umap="syllables", overlay_mode="both",
           timeline_mode="both"):
           
    timeline_track_h = 30
    zoom_track_h = 60
    margin = 15
    
    tracks_to_draw = []
    if timeline_mode in ['syllables', 'both']:
        tracks_to_draw.append(('Syllables', syllables, syllable_colors))
    if timeline_mode in ['task', 'both']:
        tracks_to_draw.append(('Task', task_data['states'], task_colors))
        
    num_tracks = len(tracks_to_draw)
    timeline_block_h = num_tracks * timeline_track_h
    zoom_block_h = num_tracks * zoom_track_h
    panel_h = timeline_block_h + zoom_block_h + (margin * 3)
    
    if umap_coords is not None:
        umap_size = h // 2 if color_umap == "both" else h
    else:
        umap_size = 0
        
    out_h = h + panel_h
    out_w = w + umap_size

    final_w = max(2, int(out_w * scale))
    final_h = max(2, int(out_h * scale))
    final_w += final_w % 2
    final_h += final_h % 2

    if color_umap == "task":
        active_color_map = task_colors
        active_state_arr = task_data['states']
    elif color_umap == "syllables":
        active_color_map = syllable_colors
        active_state_arr = syllables
    else:
        active_color_map = None
        active_state_arr = None

    full_timeline_img = np.full((timeline_block_h, out_w, 3), 50, dtype=np.uint8)
    for t_idx, (name, state_arr, cmap) in enumerate(tracks_to_draw):
        y_start = t_idx * timeline_track_h
        y_end = (t_idx + 1) * timeline_track_h
        for fi, val in enumerate(state_arr):
            if val >= 0 and val in cmap:
                x1 = int((fi / n_frames) * out_w)
                x2 = int(((fi + 1) / n_frames) * out_w)
                full_timeline_img[y_start:y_end, x1:max(x2, x1 + 1)] = cmap[val]
        
        (tw, th), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(full_timeline_img, (5, y_start + 5), (5 + tw + 10, y_start + 5 + th + 10), (0, 0, 0), -1)
        cv2.putText(full_timeline_img, name, (10, y_start + 5 + th + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    pad = 20
    if umap_coords is not None:
        if color_umap == "both":
            h_half = h // 2
            umap_bg_syl = np.full((h_half, umap_size, 3), 20, dtype=np.uint8)
            umap_bg_task = np.full((h - h_half, umap_size, 3), 20, dtype=np.uint8)
            
            for i, (ux, uy) in enumerate(umap_coords):
                s_syl = int(syllables[i])
                if s_syl >= 0 and s_syl in syllable_colors:
                    cx = int(ux * (umap_size - 2 * pad)) + pad
                    cy = int(uy * (h_half - 2 * pad)) + pad
                    cv2.circle(umap_bg_syl, (cx, cy), 1, syllable_colors[s_syl], -1)
                
                s_task = int(task_data['states'][i])
                if s_task >= 0 and s_task in task_colors:
                    cx = int(ux * (umap_size - 2 * pad)) + pad
                    cy = int(uy * ((h - h_half) - 2 * pad)) + pad
                    cv2.circle(umap_bg_task, (cx, cy), 1, task_colors[s_task], -1)
            
            umap_bg = np.vstack((umap_bg_syl, umap_bg_task))
        else:
            umap_bg = np.full((h, umap_size, 3), 20, dtype=np.uint8)
            for i, (ux, uy) in enumerate(umap_coords):
                s = int(active_state_arr[i])
                if s >= 0 and s in active_color_map:
                    cx = int(ux * (umap_size - 2 * pad)) + pad
                    cy = int(uy * (h - 2 * pad)) + pad
                    cv2.circle(umap_bg, (cx, cy), 2, active_color_map[s], -1)

    grid_cells, viz_radius = generate_grid_cells(w, h)
    use_nvenc = sys.platform == "linux" and codec.lower() == "nvenc"
    
    if use_nvenc:
        print(f"[INFO] Rendering {n_frames} frames via FFmpeg NVENC pipe ({final_w}x{final_h}) → {output_path}")
        cmd = [
            'ffmpeg', '-y',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', f'{final_w}x{final_h}',
            '-pix_fmt', 'bgr24',
            '-r', str(fps),
            '-i', '-',
            '-c:v', 'h264_nvenc',
            '-preset', 'p4', 
            '-pix_fmt', 'yuv420p',
            output_path
        ]
        writer_process = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    else:
        print(f"[INFO] Rendering {n_frames} frames via OpenCV ({codec} @ {final_w}x{final_h}) → {output_path}")
        backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" and codec.lower() in ['avc1', 'hvc1'] else cv2.CAP_ANY
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(output_path, backend, fourcc, fps, (final_w, final_h))
        
        if not writer.isOpened():
            sys.exit(f"[ERROR] cv2.VideoWriter failed to open. The codec '{codec}' is unsupported on your system or the path is invalid.")

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    for fi in range(n_frames):
        if fi % 500 == 0:
            print(f"  {fi}/{n_frames}  ({fi / n_frames * 100:.0f}%)", flush=True)

        ok, frame = cap.read()
        if not ok:
            break

        canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        canvas[:h, :w] = frame

        syl = int(syllables[fi]) if fi < len(syllables) else -1
        if color_umap != "both":
            current_state_color = active_color_map.get(int(active_state_arr[fi]), (150, 150, 150)) if active_state_arr is not None and fi < len(active_state_arr) else (150, 150, 150)

        if overlay_mode in ['task', 'both'] and fi < len(task_data['active_targets']):
            for cx_cell, cy_cell in grid_cells:
                cv2.circle(canvas, (cx_cell, cy_cell), viz_radius, (100, 100, 100), 1)

            active_t = task_data['active_targets'][fi]
            tr = int(task_data['target_radii'][fi])
            cx_mouse = task_data['cx'][fi]
            cy_mouse = task_data['cy'][fi]
            
            valid_mouse = not (math.isnan(cx_mouse) or math.isnan(cy_mouse))
            if valid_mouse:
                cx_mouse, cy_mouse = int(cx_mouse), int(cy_mouse)

            if active_t != -1 and active_t < len(grid_cells) and valid_mouse:
                tx, ty = grid_cells[active_t]
                cv2.circle(canvas, (tx, ty), viz_radius, (255, 255, 255), -1)
                cv2.circle(canvas, (tx, ty), tr, (0, 255, 0), 4)
                cv2.line(canvas, (tx, ty), (cx_mouse, cy_mouse), (0, 255, 255), 3)

            if valid_mouse:
                cv2.circle(canvas, (cx_mouse, cy_mouse), 5, (255, 255, 255), -1)

        if overlay_mode in ['syllables', 'both'] and fi < len(keypoints):
            pts = keypoints[fi]
            syl_color = syllable_colors.get(syl, (150, 150, 150))
            for (a, b) in edges:
                if a < len(pts) and b < len(pts):
                    if np.isfinite(pts[a]).all() and np.isfinite(pts[b]).all():
                        cv2.line(canvas, 
                                 (int(pts[a][0]), int(pts[a][1])), 
                                 (int(pts[b][0]), int(pts[b][1])), 
                                 syl_color, 2)

            for pt in pts:
                if np.isfinite(pt).all():
                    cv2.circle(canvas, (int(pt[0]), int(pt[1])), 4, syl_color, -1)

        if umap_coords is not None:
            canvas[:h, w:out_w] = umap_bg.copy()
            
            if color_umap == "both":
                h_half = h // 2
                cv2.putText(canvas, "UMAP: Syllables", (w + 10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                cv2.putText(canvas, "UMAP: Task", (w + 10, h_half + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                
                if fi < len(umap_coords):
                    ux, uy = umap_coords[fi]
                    
                    cx_syl = w + pad + int(ux * (umap_size - 2 * pad))
                    cy_syl = pad + int(uy * (h_half - 2 * pad))
                    c_syl = syllable_colors.get(int(syllables[fi]), (150, 150, 150)) if fi < len(syllables) else (150, 150, 150)
                    cv2.circle(canvas, (cx_syl, cy_syl), 5, (255, 255, 255), -1)
                    cv2.circle(canvas, (cx_syl, cy_syl), 3, c_syl, -1)
                    
                    cx_tsk = w + pad + int(ux * (umap_size - 2 * pad))
                    cy_tsk = h_half + pad + int(uy * ((h - h_half) - 2 * pad))
                    c_tsk = task_colors.get(int(task_data['states'][fi]), (150, 150, 150)) if fi < len(task_data['states']) else (150, 150, 150)
                    cv2.circle(canvas, (cx_tsk, cy_tsk), 5, (255, 255, 255), -1)
                    cv2.circle(canvas, (cx_tsk, cy_tsk), 3, c_tsk, -1)
            else:
                cv2.putText(canvas, "UMAP Latent Space", (w + 15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
                
                if fi < len(umap_coords):
                    ux, uy = umap_coords[fi]
                    cx_umap = w + pad + int(ux * (umap_size - 2 * pad))
                    cy_umap = pad + int(uy * (h - 2 * pad))
                    cv2.circle(canvas, (cx_umap, cy_umap), 8, (255, 255, 255), -1)
                    cv2.circle(canvas, (cx_umap, cy_umap), 5, current_state_color, -1)

        if syl >= 0 and overlay_mode in ['syllables', 'both']:
            text = f"Syllable {syl}"
            syl_color = syllable_colors.get(syl, (150, 150, 150))
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            cv2.rectangle(canvas, (10, 10), (20 + tw, 20 + th), (0, 0, 0), -1)
            cv2.putText(canvas, text, (15, 15 + th), cv2.FONT_HERSHEY_SIMPLEX, 1, syl_color, 2)

        y1_full = h + margin
        y2_full = y1_full + timeline_block_h
        canvas[y1_full:y2_full, :] = full_timeline_img
        
        cx_full = int((fi / n_frames) * out_w)
        cv2.line(canvas, (cx_full, y1_full), (cx_full, y2_full), (255, 255, 255), 2)

        y1_zoom = y2_full + margin
        y2_zoom = y1_zoom + zoom_block_h
        cv2.rectangle(canvas, (0, y1_zoom), (out_w, y2_zoom), (30, 30, 30), -1)

        half = window // 2
        t0 = max(0, fi - half)
        t1 = min(n_frames, fi + half)
        window_len = window

        if t1 > t0:
            for t_idx, (name, state_arr, cmap) in enumerate(tracks_to_draw):
                y_s = y1_zoom + t_idx * zoom_track_h
                y_e = y1_zoom + (t_idx + 1) * zoom_track_h
                win_states = state_arr[t0:t1]
                
                for i, val in enumerate(win_states):
                    if val >= 0 and val in cmap:
                        x1 = int((i / window_len) * out_w)
                        x2 = int(((i + 1) / window_len) * out_w)
                        cv2.rectangle(canvas, (x1, y_s), (max(x2, x1 + 1), y_e), cmap[val], -1)

                (tw, th), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(canvas, (5, y_s + 5), (5 + tw + 10, y_s + 5 + th + 10), (0, 0, 0), -1)
                cv2.putText(canvas, name, (10, y_s + 5 + th + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cx_zoom = int(((fi - t0) / window_len) * out_w)
        cv2.line(canvas, (cx_zoom, y1_zoom), (cx_zoom, y2_zoom), (255, 255, 255), 2)

        if (canvas.shape[1], canvas.shape[0]) != (final_w, final_h):
            frame_to_write = cv2.resize(canvas, (final_w, final_h), interpolation=cv2.INTER_AREA)
        else:
            frame_to_write = canvas

        if use_nvenc:
            writer_process.stdin.write(frame_to_write.tobytes())
        else:
            writer.write(frame_to_write)

    if use_nvenc:
        writer_process.stdin.close()
        writer_process.wait()
    else:
        writer.release()
        
    cap.release()
    print(f"[INFO] Video rendering complete.")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Overlay SLEAP keypoints, MoSeq syllables, Task Structure, and Latent UMAP on a mouse video.")
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
    
    p.add_argument("--codec", type=str, default="mp4v",
                   help="FourCC video codec. Use 'avc1' for Mac GPU, or 'nvenc' for Linux NVIDIA GPU. (default: mp4v)")
    p.add_argument("--scale", type=float, default=1.0,
                   help="Scale down output resolution. e.g., 0.5 for half resolution. (default: 1.0)")
    p.add_argument("--no-umap", action="store_true",
                   help="Disable the UMAP calculation and side-by-side visualization panel.")
    p.add_argument("--color-umap", choices=["syllables", "task", "both"], default="syllables",
                   help="Color the UMAP points by behavioral syllables, task structure, or both. (default: syllables)")
    p.add_argument("--timeline-mode", choices=["syllables", "task", "both"], default="both",
                   help="Choose which tracks to display on the bottom timeline. (default: both)")
    p.add_argument("--overlay-mode", choices=["syllables", "task", "both"], default="both",
                   help="Choose what visual elements to render onto the video frame itself. (default: both)")
                   
    return p.parse_args()


def resolve_paths(args):
    subject, session = args.subject, args.session
    data_dir, kpms_dir = Path(args.data_dir), Path(args.kpms_dir)
    model_dir, model = Path(args.model_dir), args.model_folder

    def _glob_one(root, pattern, label, required=True):
        matches = list(root.glob(pattern))
        if not matches:
            if required:
                sys.exit(f"[ERROR] No {label} file found matching: {root / pattern}")
            return None
        return str(matches[0])

    def _glob_one_flexible(root, suffix, label):
        matches = list(root.rglob(f"*{suffix}"))
        valid = [m for m in matches if subject in m.name and session in m.name]
        if not valid:
            sys.exit(f"[ERROR] No {label} file found matching subject={subject}, session={session} in {root}")
        return str(valid[0])

    slp_file  = _glob_one(model_dir, f"{model}/**/[0-9]*{subject}_{session}.slp", "SLEAP .slp")
    vid_file  = _glob_one(data_dir, f"**/[0-9]*{subject}_{session}.avi", "video .avi")

    needs_syllables = (args.color_umap in ['syllables', 'both'] and not args.no_umap) or \
                      args.overlay_mode in ['syllables', 'both'] or \
                      args.timeline_mode in ['syllables', 'both']

    kpms_file = _glob_one(kpms_dir, f"**/[0-9]*{subject}_{session}.csv", "kpms .csv", required=needs_syllables)

    eventsA_file = _glob_one_flexible(data_dir, "eventsA.csv", "eventsA")
    eventsB_file = _glob_one_flexible(data_dir, "eventsB.csv", "eventsB")
    kinematicsA_file = _glob_one_flexible(data_dir, "kinematicsA.csv", "kinematicsA")
    vid_ts_file = _glob_one_flexible(data_dir, f"{session}_video_timestamp.csv", "video_timestamp")

    out_dir = kpms_dir / "annotated_videos"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    overlay_str = f"overlay-{args.overlay_mode}"
    umap_str = "noumap" if args.no_umap else f"umap-{args.color_umap}"
    timeline_str = f"timeline-{args.timeline_mode}"
    filename = f"{subject}_{session}_annotated_{overlay_str}_{umap_str}_{timeline_str}.mp4"
    out_file = str(out_dir / filename)

    return slp_file, vid_file, kpms_file, eventsA_file, eventsB_file, kinematicsA_file, vid_ts_file, out_file


def main():
    args = parse_args()
    slp_file, vid_file, kpms_file, eventsA_file, eventsB_file, kinematicsA_file, vid_ts_file, out_file = resolve_paths(args)

    if os.path.exists(out_file):
        os.remove(out_file)

    temp_video = out_file.replace(".mp4", "_silent.mp4")
    temp_audio = out_file.replace(".mp4", "_audio.wav")

    cap, n_frames, vid_fps, h, w = open_video(vid_file)
    fps = args.fps or vid_fps

    if args.max_frames:
        n_frames = min(n_frames, args.max_frames)

    keypoints, edges, node_names = load_keypoints_kpms(
        slp_file, project_dir=args.kpms_dir,
        outlier_scale_factor=args.outlier_scale_factor)
        
    if kpms_file:
        syllables, latents = load_syllables_and_latents(kpms_file, n_frames)
    else:
        print("[INFO] No syllables file loaded. Bypassing syllable arrays.")
        syllables = np.full(n_frames, -1, dtype=int)
        latents = None

    task_data, n_frames = load_task_data(eventsA_file, eventsB_file, kinematicsA_file, vid_ts_file, n_frames)

    if len(keypoints) > n_frames:
        keypoints = keypoints[:n_frames]
    if len(syllables) > n_frames:
        syllables = syllables[:n_frames]
    if latents is not None and len(latents) > n_frames:
        latents = latents[:n_frames]

    syllable_colors = build_syllable_cmap(syllables)
    task_colors = build_task_cmap()
    
    umap_coords = None
    if not args.no_umap and latents is not None:
        if "umap" not in sys.modules:
            sys.exit("[ERROR] umap-learn is required for the latent visualizer. Run: pip install umap-learn")
        print("[INFO] Computing UMAP on latent states... This may take a moment.")
        reducer = umap.UMAP(n_components=2, random_state=42)
        u_emb = reducer.fit_transform(latents)
        
        u_min, u_max = u_emb.min(axis=0), u_emb.max(axis=0)
        umap_coords = (u_emb - u_min) / (u_max - u_min + 1e-8)

    render(cap, n_frames, h, w, keypoints, syllables, syllable_colors,
           edges, umap_coords, task_data, task_colors, temp_video, fps, 
           window=args.window, codec=args.codec, scale=args.scale, 
           color_umap=args.color_umap, overlay_mode=args.overlay_mode,
           timeline_mode=args.timeline_mode)

    print("[INFO] Generating audio track for reward states...")
    generate_reward_beeps(task_data['states'], fps, n_frames, temp_audio)

    print("[INFO] Multiplexing audio and video streams using FFmpeg...")
    mux_cmd = [
        'ffmpeg', '-y',
        '-i', temp_video,
        '-i', temp_audio,
        '-c:v', 'copy',
        '-c:a', 'aac',
        '-strict', 'experimental',
        out_file
    ]

    try:
        subprocess.run(mux_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        os.remove(temp_video)
        os.remove(temp_audio)
        print(f"[INFO] Final annotated video with audio saved → {out_file}")
    except FileNotFoundError:
        print("[WARN] FFmpeg executable not found in PATH. Leaving silent video and .wav audio files separate.")
        os.rename(temp_video, out_file)
    except subprocess.CalledProcessError:
        print("[ERROR] FFmpeg failed to multiplex streams. Leaving silent video and .wav audio files separate.")
        os.rename(temp_video, out_file)


if __name__ == "__main__":
    main()