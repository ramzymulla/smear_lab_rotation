"""
visualize_moseq.py
==================
Overlay SLEAP keypoints and keypoint-MoSeq syllable predictions on a mouse
behavior video and save the result as an MP4.

Usage
-----
python visualize_moseq.py \
    --video     path/to/video.avi \
    --slp       path/to/tracking.slp \
    --csv       path/to/syllables.csv \
    --output    path/to/output.mp4 \
    [--fps      30] \
    [--instance 0]

Dependencies
------------
    pip install opencv-python matplotlib numpy h5py pandas
    # SLEAP (for reading .slp files):
    pip install sleap  OR  conda install -c sleap sleap

Notes on .slp format
---------------------
SLEAP .slp files are HDF5 archives.  The script tries two strategies:

  1. Use sleap.load_file() if the sleap package is installed.
  2. Fall back to reading raw HDF5 keys (works when sleap is unavailable).

Both paths produce a (T, n_nodes, 2) float array of (x, y) coordinates.

Notes on syllable CSV format
-----------------------------
The CSV is expected to have at least one of:
  • A "syllable" column (one label per row == one label per frame), OR
  • "frame" and "syllable" columns (sparse/labeled rows only).

Adjust `load_syllables()` below if your CSV has a different layout.
"""

import argparse
import sys
import warnings
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")                          # no display needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.animation import FFMpegWriter
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd
import keypoint_moseq as kpms



# ─────────────────────────────────────────────────────────────────────────────
# Data loaders
# ─────────────────────────────────────────────────────────────────────────────

def open_video(video_path: str):
    """
    Open a video file and return (cap, n_frames, fps, height, width).
    The caller is responsible for cap.release().  No frames are read here,
    so memory use is negligible regardless of file size.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        sys.exit(f"[ERROR] Cannot open video: {video_path}")
    fps      = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    h        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    print(f"[INFO] Opened video: {n_frames} frames  {w}x{h}  {fps:.1f} fps  ({video_path})")
    return cap, n_frames, fps, h, w


def load_keypoints_kpms(slp_path: str, project_dir: str,
                        outlier_scale_factor: float = 6.0):
    """
    Load and clean keypoints from a .slp file using keypoint_moseq.

    Uses ``kpms.load_keypoints`` (the same loader used during model fitting)
    so the coordinate array is guaranteed to match what MoSeq was trained on.
    Outliers — keypoints whose distance to the medoid exceeds
    ``outlier_scale_factor`` standard deviations — are detected and replaced
    by linear interpolation, as is any NaN from occluded/missing nodes.

    Parameters
    ----------
    slp_path : str
        Path to the SLEAP .slp predictions file.
    project_dir : str
        kpms project directory (needed by ``outlier_removal`` for config
        and QA plot output; the directory must exist).
    outlier_scale_factor : float
        Stringency of outlier detection.  Higher → fewer points flagged.
        kpms default is 6.0.

    Returns
    -------
    coords : ndarray, shape (T, n_nodes, 2)
    edges  : list of (int, int)   – 0-indexed skeleton edge pairs
    node_names : list of str
    """

    # ── Load via kpms (handles all SLEAP format quirks internally) ────────────
    coordinates, confidences, bodyparts = kpms.load_keypoints(slp_path, "sleap")

    # load_keypoints returns a dict keyed by recording name; grab the one entry
    if len(coordinates) == 0:
        sys.exit(f"[ERROR] kpms.load_keypoints returned no data for {slp_path}")
    if len(coordinates) > 1:
        print(f"[WARN] Multiple recordings found; using the first: "
              f"{list(coordinates.keys())[0]}")
    key    = list(coordinates.keys())[0]
    coords = coordinates[key]          # (T, n_nodes, 2)
    confs  = confidences[key]          # (T, n_nodes)
    node_names = list(bodyparts)

    print(f"[INFO] Loaded keypoints via kpms  key='{key}'  shape={coords.shape}")

    # ── Outlier detection + interpolation ─────────────────────────────────────
    print(f"[INFO] Running outlier removal  (scale_factor={outlier_scale_factor})")
    coordinates_clean, confidences_clean = kpms.util.outlier_removal(
        coordinates={key: coords},
        confidences={key: confs},
        project_dir=project_dir,
        outlier_scale_factor=outlier_scale_factor,
        bodyparts=node_names,
        overwrite=True,
    )
    coords = coordinates_clean[key].astype(np.float32)
    print(f"[INFO] Outlier removal done  shape={coords.shape}")

    # ── Extract skeleton edges from the kpms config ───────────────────────────
    # kpms stores the skeleton as pairs of bodypart *names*; convert to indices
    try:
        config = kpms.load_config(project_dir)
        skeleton_pairs = config.get("skeleton", [])
        name_to_idx = {n: i for i, n in enumerate(node_names)}
        edges = [
            (name_to_idx[a], name_to_idx[b])
            for a, b in skeleton_pairs
            if a in name_to_idx and b in name_to_idx
        ]
        print(f"[INFO] Skeleton edges from kpms config: {edges}")
    except Exception as exc:
        warnings.warn(f"Could not load skeleton from kpms config ({exc}); "
                      "falling back to linear chain.")
        edges = [(0, 1), (0, 2), (0, 3), (4, 5), (4, 6), (4, 7), (4, 8), (8, 0), (7, 9)]

    return coords, edges, node_names




def load_syllables(csv_path: str, n_frames: int):
    """
    Load syllable labels and return a 1-D int array of length n_frames.
    Frames with no label get value -1.
    """
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]

    syllables = np.full(n_frames, -1, dtype=int)

    if "syllable" in df.columns and "frame" not in df.columns:
        # One row per frame (dense)
        vals = df["syllable"].values[:n_frames]
        syllables[:len(vals)] = pd.to_numeric(pd.Series(vals), errors="coerce").fillna(-1).astype(int)
    elif "frame" in df.columns and "syllable" in df.columns:
        # Sparse: only labeled frames present
        for _, row in df.iterrows():
            fi = int(row["frame"])
            if 0 <= fi < n_frames:
                syllables[fi] = int(row["syllable"])
    else:
        # Guess: first column = frame index, second = syllable
        cols = df.columns.tolist()
        print(f"[WARN] Expected 'frame'/'syllable' columns; guessing from {cols}")
        if len(cols) >= 2:
            for _, row in df.iterrows():
                fi = int(row[cols[0]])
                if 0 <= fi < n_frames:
                    syllables[fi] = int(row[cols[1]])
        elif len(cols) == 1:
            vals = df[cols[0]].values[:n_frames]
            syllables[:len(vals)] = pd.to_numeric(pd.Series(vals), errors="coerce").fillna(-1).astype(int)

    unique = sorted(set(syllables[syllables >= 0]))
    print(f"[INFO] Loaded syllables  ({csv_path})  "
          f"unique={unique[:10]}{'…' if len(unique)>10 else ''}")
    return syllables


# ─────────────────────────────────────────────────────────────────────────────
# Colour helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_syllable_cmap(syllable_ids):
    """Return a dict mapping syllable int → RGBA colour."""
    unique = sorted(set(s for s in syllable_ids if s >= 0))
    base   = matplotlib.colormaps["tab20"].resampled(max(len(unique), 1))
    return {s: base(i % 20) for i, s in enumerate(unique)}


# ─────────────────────────────────────────────────────────────────────────────
# Syllable timeline bar
# ─────────────────────────────────────────────────────────────────────────────

def build_syllable_timeline(ax_timeline, syllables, syllable_colors, n_frames, window=300):
    """
    Draw a colour-coded syllable strip in ax_timeline using a single image array.
    """
    ax_timeline.set_xlim(0, n_frames)
    ax_timeline.set_ylim(0, 1)
    ax_timeline.axis("off")

    # Create an RGBA image array for the timeline, default to grey for unlabeled frames
    timeline_img = np.zeros((1, n_frames, 4), dtype=np.float32)
    timeline_img[0, :, :] = [0.5, 0.5, 0.5, 1.0]

    # Map syllable colours onto the image array
    for syl, color in syllable_colors.items():
        timeline_img[0, syllables == syl, :] = color

    # Draw the entire timeline as one flat image
    ax_timeline.imshow(timeline_img, aspect="auto", extent=[0, n_frames, 0, 1], 
                       interpolation="nearest", zorder=1)

    # Cursor line (updated per frame)
    cursor, = ax_timeline.plot([0, 0], [0, 1], color="white",
                               linewidth=1.5, zorder=5)
    return cursor


# ─────────────────────────────────────────────────────────────────────────────
# Main render loop
# ─────────────────────────────────────────────────────────────────────────────

def render(cap, n_frames, h, w, keypoints, syllables, syllable_colors,
           edges, output_path, fps, window=300):
    """
    Render the annotated video frame-by-frame using matplotlib.
    Frames are read one at a time from `cap` so memory use stays constant
    regardless of video length — only a single decoded frame is in RAM at
    any moment.

    Layout
    ------
    ┌──────────────────────────────────────┐
    │           mouse video                │  (ax_video)
    │     with keypoints overlaid          │
    └──────────────────────────────────────┘
    ┌──────────────────────────────────────┐
    │  syllable colour timeline strip      │  (ax_timeline)
    └──────────────────────────────────────┘
    ┌──────────────────────────────────────┐
    │  zoomed timeline window ±window/2    │  (ax_zoom)
    └──────────────────────────────────────┘
    """
    # Rewind to the first frame so imshow can get an initial image
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ok, first_frame = cap.read()
    if not ok:
        sys.exit("[ERROR] Could not read first frame from video.")
    first_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
    del first_frame   # free immediately; we only needed dimensions & initial image

    fig_width  = 8
    fig_height = fig_width * h / w + 1.4   # +1.4 for the two timeline rows

    fig = plt.figure(figsize=(fig_width, fig_height), facecolor="#1a1a1a")
    gs  = fig.add_gridspec(3, 1, height_ratios=[h/w * fig_width, 0.25, 0.35],
                           hspace=0.08, left=0.04, right=0.96,
                           top=0.97, bottom=0.03)

    ax_video    = fig.add_subplot(gs[0])
    ax_timeline = fig.add_subplot(gs[1])
    ax_zoom     = fig.add_subplot(gs[2])

    ax_video.axis("off")
    ax_video.set_facecolor("#1a1a1a")

    # ── Video frame ──────────────────────────────────────────────────────────
    im = ax_video.imshow(first_rgb, aspect="auto", interpolation="bilinear")

    # ── Keypoint scatter ─────────────────────────────────────────────────────
    scat = ax_video.scatter([], [], s=25, c="cyan", zorder=4,
                             edgecolors="white", linewidths=0.5)

    # ── Skeleton lines ────────────────────────────────────────────────────────
    lc = LineCollection([], colors="lime", linewidths=1.2,
                        alpha=0.75, zorder=3)
    ax_video.add_collection(lc)

    # ── Syllable label text ───────────────────────────────────────────────────
    syl_text = ax_video.text(
        0.02, 0.97, "", transform=ax_video.transAxes,
        color="white", fontsize=11, fontweight="bold",
        va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.3", fc="#00000088", ec="none"))

    # ── Full timeline ─────────────────────────────────────────────────────────
    ax_timeline.set_facecolor("#111111")
    for spine in ax_timeline.spines.values():
        spine.set_visible(False)
    cursor_full = build_syllable_timeline(
        ax_timeline, syllables, syllable_colors, n_frames)
    ax_timeline.set_title("syllable timeline", color="#aaaaaa",
                           fontsize=7, pad=2)

    # ── Zoomed timeline ───────────────────────────────────────────────────────
    ax_zoom.set_facecolor("#111111")
    for spine in ax_zoom.spines.values():
        spine.set_color("#333333")
    ax_zoom.set_xlim(0, window)
    ax_zoom.set_ylim(0, 1)
    ax_zoom.tick_params(colors="#666666", labelsize=6)
    ax_zoom.yaxis.set_visible(False)
    ax_zoom.set_title(f"±{window//2} frame window", color="#aaaaaa",
                      fontsize=7, pad=2)

    # Zoomed timeline is redrawn from scratch each frame using BrokenBarHCollection.
    # We keep a single artist and replace its data rather than adding/removing
    # patches, which avoids both the polygon-shape error and accumulating artists.
    zoom_collection = matplotlib.collections.BrokenBarHCollection(
        [], (0, 1), facecolors=[], linewidth=0, alpha=0.9)
    ax_zoom.add_collection(zoom_collection)
    cursor_zoom, = ax_zoom.plot([window // 2, window // 2], [0, 1],
                                 color="white", linewidth=1.5, zorder=5)

    # ── Legend ────────────────────────────────────────────────────────────────
    handles = [mpatches.Patch(color=c, label=f"Syl {s}")
               for s, c in sorted(syllable_colors.items())]
    if handles:
        ax_video.legend(handles=handles[:12],   # cap at 12 to avoid overflow
                        loc="upper right", fontsize=6,
                        framealpha=0.5, facecolor="#111111",
                        labelcolor="white", ncol=2,
                        borderpad=0.4, handlelength=1)

    # ── FFmpeg writer ─────────────────────────────────────────────────────────
    writer = FFMpegWriter(fps=fps, bitrate=4000,
                          extra_args=["-vcodec", "libx264",
                                      "-pix_fmt", "yuv420p"])

    # Rewind before the render loop (we already read frame 0 above)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    print(f"[INFO] Rendering {n_frames} frames → {output_path}")
    with writer.saving(fig, output_path, dpi=120):
        for fi in range(n_frames):
            if fi % 200 == 0:
                pct = fi / n_frames * 100
                print(f"  {fi}/{n_frames}  ({pct:.0f}%)", flush=True)

            # ── Read next frame from disk (one at a time — no bulk storage) ───
            ok, raw = cap.read()
            if not ok:
                print(f"[WARN] Video ended early at frame {fi}; stopping.")
                break
            frame_rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
            del raw  # release the BGR copy immediately

            # ── Update video frame ────────────────────────────────────────────
            im.set_data(frame_rgb)
            del frame_rgb  # release the RGB copy immediately

            # ── Update keypoints ──────────────────────────────────────────────
            if fi < len(keypoints):
                pts = keypoints[fi]                 # (n_nodes, 2)
                valid = np.isfinite(pts).all(axis=1)
                scat.set_offsets(pts[valid])

                # skeleton edges
                segs = []
                for (a, b) in edges:
                    if a < len(pts) and b < len(pts):
                        if np.isfinite(pts[a]).all() and np.isfinite(pts[b]).all():
                            segs.append([pts[a], pts[b]])
                lc.set_segments(segs)
            else:
                scat.set_offsets(np.empty((0, 2)))
                lc.set_segments([])

            # ── Update syllable label ─────────────────────────────────────────
            syl = int(syllables[fi]) if fi < len(syllables) else -1
            if syl >= 0:
                col = syllable_colors.get(syl, "white")
                syl_text.set_text(f"Syllable  {syl}")
                syl_text.set_bbox(dict(boxstyle="round,pad=0.3",
                                       fc=(*col[:3], 0.55), ec="none"))
            else:
                syl_text.set_text("")

            # ── Update full-timeline cursor ───────────────────────────────────
            cursor_full.set_xdata([fi, fi])

            # ── Update zoomed timeline ────────────────────────────────────────
            half = window // 2
            t0 = max(0, fi - half)
            t1 = min(n_frames, fi + half)

            ax_zoom.set_xlim(t0, t0 + window)
            ax_zoom.set_xticks(
                np.linspace(t0, min(t0 + window, n_frames - 1), 5, dtype=int))

            # Build (xstart, width) spans and matching colours for every
            # contiguous run of the same syllable in the visible window.
            # BrokenBarHCollection accepts exactly this format — no polygon
            # vertices, no None sentinels, no shape ambiguity.
            window_syls = syllables[t0:t1]
            xranges, facecolors = [], []
            run_start = t0
            run_syl   = window_syls[0] if len(window_syls) else -1
            for i, s in enumerate(window_syls):
                if s != run_syl:
                    if run_syl >= 0:
                        xranges.append((run_start, t0 + i - run_start))
                        facecolors.append(syllable_colors.get(run_syl, "grey"))
                    run_start = t0 + i
                    run_syl   = s
            # close the last run
            if run_syl >= 0 and len(window_syls):
                xranges.append((run_start, t0 + len(window_syls) - run_start))
                facecolors.append(syllable_colors.get(run_syl, "grey"))

            # cursor_zoom is a Line2D in ax_zoom.lines — clearing
            # ax_zoom.collections only removes patch/collection artists
            # so the cursor is untouched.
            # cursor_zoom is a Line2D in ax_zoom.lines — clearing
            # ax_zoom.collections only removes patch/collection artists
            # so the cursor is untouched.
            for coll in list(ax_zoom.collections):
                coll.remove()
            
            if xranges:
                zoom_collection = matplotlib.collections.BrokenBarHCollection(
                    xranges, (0, 1), facecolors=facecolors,
                    linewidth=0, alpha=0.9)
                ax_zoom.add_collection(zoom_collection)

            cursor_zoom.set_xdata([fi, fi])

            writer.grab_frame()

    cap.release()
    plt.close(fig)
    print(f"[INFO] Done → {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Overlay SLEAP keypoints + MoSeq syllables on a mouse video.")

    # ── Identity ──────────────────────────────────────────────────────────────
    p.add_argument("--subject",  required=True,
                   help="Subject ID  (used to glob for input files)")
    p.add_argument("--session",  required=True,
                   help="Session ID  (used to glob for input files)")

    # ── Directory / project overrides ─────────────────────────────────────────
    p.add_argument("--kpms-dir",
                   default="/Users/ramzyalmulla/research/smearlab/kpms",
                   help="Root dir for kpms outputs & annotated_videos/")
    p.add_argument("--data-dir",
                   default="/Volumes/CrucialRZA/smearlab/clickbait-loco/thermister/",
                   help="Root dir containing .avi and .slp files")
    p.add_argument("--model-folder",
                   default="labels_v003_0.5scaling",
                   help="Sub-folder under data-dir that contains .slp files")

    # ── Optional overrides ────────────────────────────────────────────────────
    p.add_argument("--fps",        type=float, default=None,
                   help="Override output FPS  (default: use video FPS)")
    p.add_argument("--window",     type=int,   default=300,
                   help="Frame width of the zoomed timeline  (default: 300)")
    p.add_argument("--max-frames", type=int,   default=None,
                   help="Truncate to this many frames (useful for testing)")
    p.add_argument("--outlier-scale-factor", type=float, default=6.0,
                   help="Outlier detection stringency for kpms outlier_removal "
                        "(higher -> fewer points flagged, default: 6.0)")
    return p.parse_args()


def resolve_paths(args):
    """
    Use the lab's standard naming convention to locate all inputs and build
    the output path — mirroring the snippet in the project README.

    Returns (slp_file, vid_file, kpms_file, out_file) as plain strings.
    """
    subject = args.subject
    session = args.session
    data_dir = Path(args.data_dir)
    kpms_dir = Path(args.kpms_dir)
    model    = args.model_folder

    def _glob_one(root, pattern, label):
        matches = list(root.glob(pattern))
        if not matches:
            sys.exit(f"[ERROR] No {label} file found matching: {root / pattern}")
        if len(matches) > 1:
            print(f"[WARN] Multiple {label} files found; using the first:")
            for m in matches:
                print(f"  {m}")
        return str(matches[0])

    slp_file  = _glob_one(data_dir, f"{model}/**/[0-9]*{subject}_{session}.slp", "SLEAP .slp")
    vid_file  = _glob_one(data_dir, f"**/[0-9]*{subject}_{session}.avi",         "video .avi")
    kpms_file = _glob_one(kpms_dir, f"**/[0-9]*{subject}_{session}.csv",         "kpms .csv")

    out_dir = kpms_dir / "annotated_videos"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = str(out_dir / f"{subject}_{session}_annotated.mp4")

    print(f"[INFO] SLEAP  : {slp_file}")
    print(f"[INFO] Video  : {vid_file}")
    print(f"[INFO] KPMS   : {kpms_file}")
    print(f"[INFO] Output : {out_file}")
    return slp_file, vid_file, kpms_file, out_file

import cv2
import numpy as np

def build_syllable_cmap(syllable_ids):
    """Return a dict mapping syllable int → BGR color tuple for OpenCV."""
    import matplotlib
    unique = sorted(set(s for s in syllable_ids if s >= 0))
    base = matplotlib.colormaps["tab20b"].resampled(max(len(unique), 1))
    colors = {}
    for i, s in enumerate(unique):
        r, g, b, a = base(i % 20)
        colors[s] = (int(b * 255), int(g * 255), int(r * 255))  # Convert to BGR
    return colors

def render(cap, n_frames, h, w, keypoints, syllables, syllable_colors, edges, output_path, fps, window=300):
    # Layout configuration
    timeline_h = 30
    zoom_h = 60
    margin = 15
    panel_h = timeline_h + zoom_h + (margin * 3)
    out_h = h + panel_h
    out_w = w

    # Pre-render the full timeline array to avoid calculating it per-frame
    full_timeline_img = np.full((timeline_h, out_w, 3), 50, dtype=np.uint8)
    for fi, syl in enumerate(syllables):
        if syl >= 0 and syl in syllable_colors:
            x1 = int((fi / n_frames) * out_w)
            x2 = int(((fi + 1) / n_frames) * out_w)
            full_timeline_img[:, x1:max(x2, x1 + 1)] = syllable_colors[syl]

    # Initialize VideoWriter
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

        # Draw Skeleton Edges
        if fi < len(keypoints):
            pts = keypoints[fi]
            for (a, b) in edges:
                if a < len(pts) and b < len(pts):
                    if np.isfinite(pts[a]).all() and np.isfinite(pts[b]).all():
                        cv2.line(canvas, 
                                 (int(pts[a][0]), int(pts[a][1])), 
                                 (int(pts[b][0]), int(pts[b][1])), 
                                 (0, 255, 0), 2)

            # Draw Keypoints
            for pt in pts:
                if np.isfinite(pt).all():
                    cv2.circle(canvas, (int(pt[0]), int(pt[1])), 4, (255, 255, 0), -1)

        # Draw Syllable Text
        syl = int(syllables[fi]) if fi < len(syllables) else -1
        if syl >= 0:
            color = syllable_colors.get(syl, (255, 255, 255))
            text = f"Syllable {syl}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            cv2.rectangle(canvas, (10, 10), (20 + tw, 20 + th), (0, 0, 0), -1)
            cv2.putText(canvas, text, (15, 15 + th), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # Draw Full Timeline
        y1_full = h + margin
        y2_full = y1_full + timeline_h
        canvas[y1_full:y2_full, :] = full_timeline_img
        
        # Cursor for full timeline
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
                    # Map the relative window position to the full width of the video
                    x1 = int((i / window_len) * out_w)
                    x2 = int(((i + 1) / window_len) * out_w)
                    cv2.rectangle(canvas, (x1, y1_zoom), (x2, y2_zoom), syllable_colors[s], -1)

        # Cursor for zoomed timeline (always dead center unless near edges)
        cx_zoom = int(((fi - t0) / window_len) * out_w)
        cv2.line(canvas, (cx_zoom, y1_zoom), (cx_zoom, y2_zoom), (255, 255, 255), 2)

        writer.write(canvas)

    writer.release()
    cap.release()
    print(f"[INFO] Done → {output_path}")

def main():
    args = parse_args()
    slp_file, vid_file, kpms_file, out_file = resolve_paths(args)

    # ── Open video (no frames loaded into RAM) ───────────────────────────────
    cap, n_frames, vid_fps, h, w = open_video(vid_file)
    fps = args.fps or vid_fps

    if args.max_frames:
        n_frames = min(n_frames, args.max_frames)

    # ── Load keypoints and syllables ──────────────────────────────────────────
    keypoints, edges, node_names = load_keypoints_kpms(
        slp_file, project_dir=args.kpms_dir,
        outlier_scale_factor=args.outlier_scale_factor)
    syllables = load_syllables(kpms_file, n_frames)

    # Trim keypoints to video length
    if len(keypoints) > n_frames:
        keypoints = keypoints[:n_frames]

    # ── Build colour map ──────────────────────────────────────────────────────
    syllable_colors = build_syllable_cmap(syllables)

    # ── Render (streams video frames one at a time) ───────────────────────────
    render(cap, n_frames, h, w, keypoints, syllables, syllable_colors,
           edges, out_file, fps, window=args.window)


if __name__ == "__main__":
    main()