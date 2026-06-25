import keypoint_moseq as kpms
import matplotlib.pyplot as plt
import os
import re
import fnmatch
import sys
import numpy as np
from pathlib import Path
import jax_moseq
import matplotlib as mpl
import umap

framerate = sys.argv[1]
sleap_dir = sys.argv[2]
includeStr = sys.argv[3].split('/')[-1]

def generate_session_umaps(results: dict, project_dir: str, model_name: str):
    """
    Computes and saves a static 2D UMAP scatter plot for each session 
    in the keypoint-MoSeq results dictionary.
    """
    out_dir = Path(project_dir) / model_name / "umaps" 
    out_dir.mkdir(parents=True, exist_ok=True)

    
    latentsAll = np.concatenate([results[key]['latent_state'] for key in results],axis=0)
    nFramesEachSession = np.array([results[key]['latent_state'].shape[0] for key in results])
    cumFramesEachSession = np.array([0]+[np.sum(nFramesEachSession[:i+1]) for i in range(len(nFramesEachSession)-1)])

    print(f"[INFO] Computing UMAP ({len(latentsAll)} frames)...")
    reducer = umap.UMAP(n_components=2, random_state=42)
    u_emb = reducer.fit_transform(latentsAll)
    ylims = [np.min(u_emb[:,1]),np.max(u_emb[:,1])]
    xlims = [np.min(u_emb[:,0]),np.max(u_emb[:,0])]
    
    for inds, (session_key, data) in enumerate(results.items()):
        if 'latent_state' not in data or 'syllable' not in data:
            print(f"[WARN] Missing latents or syllables for {session_key}. Skipping.")
            continue
            
        syllables = data['syllable']
        u_embThisSession = u_emb[cumFramesEachSession[inds]:cumFramesEachSession[inds]+nFramesEachSession[inds],:]
        
        
        plt.figure(figsize=(10, 8), facecolor="white")
        unique_syls = np.unique(syllables)
        unique_syls = unique_syls[unique_syls >= 0]
        
        colors = np.vstack([mpl.colormaps['tab20'].colors,
                            mpl.colormaps['tab20b'].colors,
                            mpl.colormaps['Pastel1'].colors,
                            mpl.colormaps['Pastel2'].colors])
        
        cmap = mpl.colors.ListedColormap(colors)
        

        scatter = plt.scatter(
            u_embThisSession[:, 0], 
            u_embThisSession[:, 1], 
            c= syllables, 
            cmap=cmap, 
            s=8, 
            vmin=0,
            vmax=colors.shape[0],
            alpha=0.7, 
            edgecolors="none"
        )
        plt.xlim(xlims)
        plt.ylim(ylims)
        
        # Pass unique_syls to the 'num' parameter to force a handle for every syllable
        handles, _ = scatter.legend_elements(prop="colors", alpha=1, num=len(unique_syls))
        labels = [f"Syl {s}" for s in unique_syls]
        
        maxN=colors.shape[0]
        if len(handles) > maxN:
            handles, labels = handles[:maxN], labels[:maxN]
            labels[-1] = "..."
            
            
        plt.legend(handles, labels, loc="upper right", bbox_to_anchor=(1.15, 1), fontsize=8, ncol=1)
        plt.title(f"UMAP Latent Space - {session_key}")
        plt.axis("off")
        plt.tight_layout()
        
        out_file = out_dir / f"{session_key}_umap.png"
        plt.savefig(out_file, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[INFO] Saved {out_file}")

project_dir = str(Path(f"/home/rza/Research/smearlab/scripts/kpms_outputs/rslds_keypoint_moseq_30/"))
config = lambda: kpms.load_config(project_dir)
base_path_str = "/home/rza/Research/smearlab/clickbait-loco/thermister/"
model_name = '2026_06_07-22_24_34'
sleap_files = [str(i) for i in list(Path(sleap_dir).glob(includeStr.replace('.avi','.slp')))]
print(sleap_files)
sleap_file = sleap_files[0]  # any .slp or .h5 file with predictions for a single video

video_dir = str(Path(base_path_str))  # any directory containing videos (e.g. .mp4, .avi) and/or sleap files with predictions for those videos
if not os.path.exists(project_dir):
    kpms.setup_project(project_dir, sleap_file=sleap_file,video_dir=video_dir) 

kpms.update_config(project_dir,
                    use_bodyparts=['nose1','neck1','centroid1','forepawL1','forepawR1','hindpawL1','hindpawR1','tailstart1'],
                    anterior_bodyparts=["nose1","neck1"],posterior_bodyparts=["tailstart1"],fps=int(framerate),
                    video_dir=video_dir)
model = kpms.load_checkpoint(project_dir, model_name)[0]



for subject in ['2208','2209','2210']:
# load data 
    if subject not in includeStr:
        continue

    keypoint_data_path = sleap_files # can be a file, a directory, or a list of files
    
    outDir = os.path.join(project_dir,model_name,subject)
    if not os.path.exists(outDir):
        os.mkdir(outDir)

    coordinates, confidences, bodyparts = kpms.load_keypoints(keypoint_data_path, "sleap")
    coordinates, confidences = kpms.outlier_removal(
        coordinates,
        confidences,
        project_dir,
        overwrite=False,
        **config()
    )
    

    data, metadata = kpms.format_data(coordinates, confidences, **config())
    data = jax_moseq.utils.debugging.convert_data_precision(data)

    

    # apply saved model to new data
    results = kpms.apply_model(model, data, metadata, project_dir, model_name, overwrite=True, **config())

    # results = kpms.load_results(project_dir, model_name)

    kpms.save_results_as_csv(results, outDir, model_name)

    kpms.generate_trajectory_plots(coordinates, results, outDir, model_name, **config())

    try:
        kpms.plot_similarity_dendrogram(coordinates, results, outDir, model_name, outDir **config())
    except:
        print("Failed to plot dendrogram, increasing min_frequency threshold")
        kpms.plot_similarity_dendrogram(coordinates, results, outDir, model_name, min_frequency=0.01, **config())

    generate_session_umaps(results,outDir,model_name)

    kpms.generate_grid_movies(results, outDir, model_name, 
                            overlay_keypoints=True,coordinates=coordinates, **config());
