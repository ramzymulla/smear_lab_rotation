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


rslds=True
if rslds:
    from jax_moseq.models.keypoint_slds.decision_list import greedy_decision_list_permutation, permute_model_states
    # project_dir = str(Path(f"/home/rza/Research/smearlab/scripts/kpms_outputs/rsdls_keypoint_moseq_{framerate}/"))
    # project_dir = str(Path(f"/home/rza/Research/smearlab/scripts/kpms_outputs/rslds_keypoint_moseq_{framerate}b/"))
    jitterAR=1e-1
    jitterSLDS=1e-1
    kappaAR=1e6
    kappaSLDS=1e6
else:
    # project_dir = str(Path(f"/home/rza/Research/smearlab/scripts/kpms_outputs/keypoint_moseq_{framerate}/"))
    jitterAR=1e-1
    jitterSLDS=1e-1
    kappaAR=1e10
    kappaSLDS=5e3

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

generate_session_umaps(results,project_dir,model_name)