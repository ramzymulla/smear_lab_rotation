import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
import keypoint_moseq as kpms
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
from pathlib import Path
import jax_moseq
import umap
import matplotlib as mpl


project_dir = sys.argv[1]
model_name = sys.argv[2]

framerate = 30
sleap_dir = '/home/rza/Research/smearlab/scripts/sleap/labels_v003_0.5scaling/models/260419_223138.0.5_scaling.single_instance.n=193/preds/'
includeStr = "**/*a22.avi".split('/')[-1]

def generate_session_umaps(results: dict, project_dir: str, model_name: str):
    out_dir = Path(project_dir) / "annotated_videos" / model_name / "umaps"
    out_dir.mkdir(parents=True, exist_ok=True)

    states_dict = results.get("states", {})
    latent_key = next((k for k in ["latents", "x", "v"] if k in results), None)

    if not latent_key:
        print("[ERROR] No latent states found in results dictionary.")
        return

    for session_key, syllables in states_dict.items():
        if session_key not in results[latent_key]:
            print(f"[WARN] No latents found for session {session_key}, skipping.")
            continue

        latents = results[latent_key][session_key]
        
        # Filter out unlabeled frames
        valid = syllables >= 0
        if not np.any(valid):
            print(f"[WARN] No valid syllables for session {session_key}, skipping.")
            continue

        valid_latents = latents[valid]
        valid_syllables = syllables[valid]

        print(f"[INFO] Computing UMAP for session: {session_key} ({len(valid_latents)} frames)")
        reducer = umap.UMAP(n_components=2, random_state=42)
        u_emb = reducer.fit_transform(valid_latents)

        plt.figure(figsize=(8, 8), facecolor="white")
        unique_syls = np.unique(valid_syllables)
        cmap = plt.cm.get_cmap("tab20b", max(len(unique_syls), 1))

        scatter = plt.scatter(
            u_emb[:, 0], 
            u_emb[:, 1], 
            c=valid_syllables, 
            cmap=cmap, 
            s=10, 
            alpha=0.7, 
            edgecolors="none"
        )

        handles, _ = scatter.legend_elements(prop="colors", alpha=1)
        labels = [f"Syl {s}" for s in unique_syls]

        if len(handles) > 20:
            handles, labels = handles[:20], labels[:20]
            labels[-1] = "..."

        plt.legend(handles, labels, loc="upper right", bbox_to_anchor=(1.15, 1), fontsize=8, ncol=1)
        plt.title(f"UMAP Latent Space - {session_key}")
        plt.axis("off")
        plt.tight_layout()

        out_path = out_dir / f"{session_key}_umap.png"
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[INFO] Saved -> {out_path}")

rslds=True
if rslds:
    from jax_moseq.models.keypoint_slds.decision_list import greedy_decision_list_permutation, permute_model_states
    # project_dir = str(Path(f"/home/rza/Research/smearlab/scripts/kpms_outputs/rsdls_keypoint_moseq_{framerate}/"))
    # project_dir = str(Path(f"/home/rza/Research/smearlab/scripts/kpms_outputs/rslds_keypoint_moseq_{framerate}b/"))
    jitterAR=1e-1
    jitterSLDS=1e-1
    kappaAR=1e6
    kappaSLDS=1e6
    subset = "2208"
else:
    # project_dir = str(Path(f"/home/rza/Research/smearlab/scripts/kpms_outputs/keypoint_moseq_{framerate}/"))
    jitterAR=1e-1
    jitterSLDS=1e-1
    kappaAR=1e10
    kappaSLDS=5e3

import plotly.express as px
import pandas as pd
import gc

def generate_session_umaps(results: dict, project_dir: str, model_name: str, max_3d_points: int = 50000, subset=None):
    """
    Computes and saves a static 2D UMAP scatter plot and an interactive 3D UMAP plot 
    for each session in the keypoint-MoSeq results dictionary.
    """
    print(f"[INFO] Loading latents of UMAP visualization...")
    out_dir = Path(project_dir) / model_name / "umaps" 
    out_dir.mkdir(parents=True, exist_ok=True)

    if subset is not None:
        results = {key:results[key] for key in results if subset in key}
    
    latentsAll = np.concatenate([results[key]['latent_state'] for key in results], axis=0)
    nFramesEachSession = np.array([results[key]['latent_state'].shape[0] for key in results])
    cumFramesEachSession = np.insert(np.cumsum(nFramesEachSession), 0, 0)

    print(f"[INFO] Computing 2D UMAP ({len(latentsAll)} frames)...")
    reducer_2d = umap.UMAP(n_components=2, random_state=42, verbose=True,n_jobs=-1)
    u_emb_2d = reducer_2d.fit_transform(latentsAll)
    
    print(f"[INFO] Computing 3D UMAP ({len(latentsAll)} frames)...")
    reducer_3d = umap.UMAP(n_components=3, random_state=42,verbose=True,n_jobs=-1)
    u_emb_3d = reducer_3d.fit_transform(latentsAll)
    
    xlims = [np.min(u_emb_2d[:,0]), np.max(u_emb_2d[:,0])]
    ylims = [np.min(u_emb_2d[:,1]), np.max(u_emb_2d[:,1])]

    # # Explicitly clear massive arrays and run garbage collection
    # del latentsAll
    # gc.collect()
    
    colors = np.vstack([mpl.colormaps['tab20'].colors,
                            mpl.colormaps['tab20b'].colors,
                            mpl.colormaps['Pastel1'].colors,
                            mpl.colormaps['Pastel2'].colors,
                            mpl.colormaps['Set2'].colors,
                            mpl.colormaps['Set3'].colors])
    cmap = mpl.colors.ListedColormap(colors)
    
    for inds, (session_key, data) in enumerate(results.items()):
        if 'latent_state' not in data or 'syllable' not in data:
            print(f"[WARN] Missing latents or syllables for {session_key}. Skipping.")
            continue
            
        syllables = data['syllable']
        start_idx = cumFramesEachSession[inds]
        end_idx = start_idx + nFramesEachSession[inds]
        
        u_embThisSession_2d = u_emb_2d[start_idx:end_idx, :]
        u_embThisSession_3d = u_emb_3d[start_idx:end_idx, :]
        
        # Add these lines to align array lengths
        min_len = min(len(u_embThisSession_2d), len(syllables))
        u_embThisSession_2d = u_embThisSession_2d[:min_len]
        u_embThisSession_3d = u_embThisSession_3d[:min_len]
        syllables = syllables[:min_len]
        
        unique_syls = np.unique(syllables)
        unique_syls = unique_syls[unique_syls >= 0]
        
        # --- 2D Matplotlib Plot ---
        plt.figure(figsize=(12, 8), facecolor="white")
        scatter = plt.scatter(
            u_embThisSession_2d[:, 0], 
            u_embThisSession_2d[:, 1], 
            c=syllables, 
            cmap=cmap, 
            s=8, 
            vmin=0,
            vmax=colors.shape[0],
            alpha=0.7, 
            edgecolors="none"
        )
        plt.xlim(xlims)
        plt.ylim(ylims)
        
        handles, _ = scatter.legend_elements(prop="colors", alpha=1, num=len(unique_syls))
        labels = [f"Syl {s}" for s in unique_syls]
        
        maxN = colors.shape[0]
        if len(handles) > maxN:
            handles, labels = handles[:maxN], labels[:maxN]
            labels[-1] = "..."
            
        plt.legend(handles, labels, loc="upper right", bbox_to_anchor=(1.15, 1), fontsize=8, ncol=2)
        plt.title(f"UMAP Latent Space - {session_key}")
        plt.axis("off")
        plt.tight_layout()
        plt.gcf().subplots_adjust(right=0.9)
        
        out_file_2d = out_dir / f"{session_key}_umap.png"
        plt.savefig(out_file_2d, dpi=300, bbox_inches="tight")
        plt.close()
        
        # --- 3D Interactive Plot ---
        # Subsample to prevent WebGL/RAM crashes in browser for massive sessions
        n_frames = len(u_embThisSession_3d)
        if n_frames > max_3d_points:
            step = n_frames // max_3d_points
            u_plot_3d = u_embThisSession_3d[::step]
            syl_plot = syllables[::step]
        else:
            u_plot_3d = u_embThisSession_3d
            syl_plot = syllables

        df_3d = pd.DataFrame({
            'UMAP 1': u_plot_3d[:, 0],
            'UMAP 2': u_plot_3d[:, 1],
            'UMAP 3': u_plot_3d[:, 2],
            'Syllable': syl_plot.astype(str) 
        })
        
        # Plotly Express natively handles discrete coloring when passed strings
        fig = px.scatter_3d(
            df_3d, x='UMAP 1', y='UMAP 2', z='UMAP 3', 
            color='Syllable', opacity=0.7
        )
        fig.update_traces(marker=dict(size=3))
        fig.update_layout(
            title=f"3D UMAP - {session_key}", 
            margin=dict(l=0, r=0, b=0, t=30)
        )
        
        out_file_3d = out_dir / f"{session_key}_umap_3d.html"
        # Exporting with CDN prevents a 3MB javascript bundle from inflating every HTML file
        fig.write_html(out_file_3d, include_plotlyjs='cdn')
        
        del fig
        gc.collect()

        print(f"[INFO] Saved 2D and 3D umaps for {session_key}")

base_path_str = "/home/rza/Research/smearlab/clickbait-loco/thermister/"

models_dir = "/home/rza/Research/smearlab/scripts/sleap/"

sleap_files = [str(i) for i in list(Path(sleap_dir).glob(includeStr.replace('.avi','.slp')))]
video_dir = str(Path(base_path_str))  # any directory containing videos (e.g. .mp4, .avi) and/or sleap files with predictions for those videos
if not os.path.exists(project_dir):
    kpms.setup_project(project_dir, sleap_file=sleap_files[0],video_dir=video_dir) 
else:
    kpms.update_config(project_dir,
                       sleap_file=sleap_files[0],
                       video_dir=video_dir)
    
config = lambda: kpms.load_config(project_dir)

# load data 
keypoint_data_path = sleap_files  # can be a file, a directory, or a list of files
coordinates, confidences, bodyparts = kpms.load_keypoints(keypoint_data_path, "sleap")

coordinates, confidences = kpms.outlier_removal(
    coordinates,
    confidences,
    project_dir,
    overwrite=False,
    **config()
)

# # load the most recent model checkpoint
# model, data, metadata, current_iter = kpms.load_checkpoint(project_dir, model_name)

# extract results
results = kpms.load_results(project_dir, model_name)

# kpms.save_results_as_csv(results, project_dir, model_name)

# kpms.generate_trajectory_plots(coordinates, results, project_dir, model_name, **config())

# try:
#     kpms.plot_similarity_dendrogram(coordinates, results, project_dir, model_name, **config())
# except:
#     kpms.plot_similarity_dendrogram(coordinates, results, project_dir, model_name, min_frequency=0.01, **config())
# kpms.generate_grid_movies(results, project_dir, model_name, 
#                           overlay_keypoints=True,coordinates=coordinates, **config());

generate_session_umaps(results,project_dir,model_name,subset=subset)