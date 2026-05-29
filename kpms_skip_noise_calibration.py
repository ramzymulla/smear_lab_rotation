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

framerate = sys.argv[1]
sleap_dir = sys.argv[2]
includeStr = sys.argv[3].split('/')[-1]


rslds=True if 'rslds' in sys.argv else False
if rslds:
    from jax_moseq.models.keypoint_slds.decision_list import greedy_decision_list_permutation, permute_model_states
    project_dir = str(Path(f"/home/rza/Research/smearlab/scripts/kpms_outputs/rslds_keypoint_moseq_{framerate}/"))
    jitterAR=1e-2
    jitterSLDS=1e-2
    kappaAR=1e10
    kappaSLDS=1e3
    lW = 0.1
    lb = 0.05
else:
    project_dir = str(Path(f"/home/rza/Research/smearlab/scripts/kpms_outputs/keypoint_moseq_{framerate}/"))
    jitterAR=1e-1
    jitterSLDS=1e-1
    kappaAR=1e12
    kappaSLDS=5e6
    lW = 0.1
    lb = 0.05

base_path_str = "/home/rza/Research/smearlab/clickbait-loco/thermister/"
model_name = f'kpms_model_{framerate}fps'
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


# update config
kpms.update_config(project_dir,
                   use_bodyparts=['nose1','neck1','centroid1','forepawL1','forepawR1','hindpawL1','hindpawR1','tailstart1'],
                   anterior_bodyparts=["nose1","neck1"],posterior_bodyparts=["tailstart1"],fps=int(framerate))


### PREPROCESSING ###
coordinates, confidences = kpms.outlier_removal(
    coordinates,
    confidences,
    project_dir,
    overwrite=False,
    **config()
)

data, metadata = kpms.format_data(coordinates, confidences, **config())
pca = kpms.fit_pca(**data, **config())

kpms.save_pca(pca, project_dir)

cs = np.cumsum(pca.explained_variance_ratio_)

latent_dim = min((cs>0.9).nonzero()[0].min()+1,10)

kpms.update_config(project_dir, latent_dim=int(latent_dim))
print(f"latent dim = {latent_dim}")

# kpms.update_config(
#     project_dir,
#     sigmasq_loc=kpms.estimate_sigmasq_loc(data["Y"], data["mask"], filter_size=config()["fps"])
# )
data = jax_moseq.utils.debugging.convert_data_precision(data)


### MODEL FITTING ###

# initialize the model
model = kpms.init_model(data, pca=pca, **config())



# # optionally modify kappa
model = kpms.update_hypparams(model, kappa=kappaAR)

num_ar_iters = 50

model, model_name = kpms.fit_model(
    model, data, metadata, project_dir, jitter=jitterAR,ar_only=True, num_iters=num_ar_iters
)

# load model checkpoint
model, data, metadata, current_iter = kpms.load_checkpoint(
    project_dir, model_name, iteration=num_ar_iters
)

if rslds: 
    # Extract optimal ordering and permute model based on settled SLDS latents
    num_states = model["hypparams"]["trans_hypparams"]["num_states"]
    # optimal_permutation = greedy_decision_list_permutation(
    #     model["states"]["x"], 
    #     model["states"]["z"], 
    #     num_states
    # )
    # model = permute_model_states(model, optimal_permutation)

    optimal_permutation, R_init, r_init = greedy_decision_list_permutation(
        model["states"]["x"], 
        model["states"]["z"], 
        num_states
    )
    model = permute_model_states(model, optimal_permutation, R_init, r_init)


# # modify kappa to maintain the desired syllable time-scale
model = kpms.update_hypparams(model, kappa=kappaSLDS, lambda_W = lW, lambda_b = lb)


# run SLDS/rSLDS fitting for an additional 500 iters
model, model_name = kpms.fit_model(
    model,
    data,
    metadata,
    project_dir,
    jitter=jitterSLDS,
    ar_only=False,
    start_iter=num_ar_iters,
    num_iters=current_iter+500
)


### VISUALIZE & SAVE MODEL FIT RESULTS ###

# modify a saved checkpoint so syllables are ordered by frequency
kpms.reindex_syllables_in_checkpoint(project_dir, model_name);

# load the most recent model checkpoint
model, data, metadata, current_iter = kpms.load_checkpoint(project_dir, model_name)

# extract results
results = kpms.extract_results(model, metadata, project_dir, model_name)

kpms.save_results_as_csv(results, project_dir, model_name)

kpms.generate_trajectory_plots(coordinates, results, project_dir, model_name, **config())

try:
    kpms.plot_similarity_dendrogram(coordinates, results, project_dir, model_name, **config())
except:
    kpms.plot_similarity_dendrogram(coordinates, results, project_dir, model_name, min_frequency=0.01, **config())

kpms.generate_grid_movies(results, project_dir, model_name, 
                          overlay_keypoints=True,coordinates=coordinates, **config());

