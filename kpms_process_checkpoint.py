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

model_name = sys.argv[1]

framerate = 30
sleap_dir = '/home/rza/Research/smearlab/scripts/sleap/labels_v003_0.5scaling/models/260419_223138.0.5_scaling.single_instance.n=193/preds/'
includeStr = "**/*a22.avi".split('/')[-1]


rslds=True
if rslds:
    from jax_moseq.models.keypoint_slds.decision_list import greedy_decision_list_permutation, permute_model_states
    # project_dir = str(Path(f"/home/rza/Research/smearlab/scripts/kpms_outputs/rsdls_keypoint_moseq_{framerate}/"))
    project_dir = str(Path(f"/home/rza/Research/smearlab/scripts/kpms_outputs/rslds_keypoint_moseq_{framerate}b/"))
    jitterAR=1e-1
    jitterSLDS=1e-1
    kappaAR=1e13
    kappaSLDS=1e6
else:
    project_dir = str(Path(f"/home/rza/Research/smearlab/scripts/kpms_outputs/keypoint_moseq_{framerate}/"))
    jitterAR=1e-1
    jitterSLDS=1e-1
    kappaAR=1e10
    kappaSLDS=5e3

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
kpms.generate_grid_movies(results, project_dir, model_name, 
                          overlay_keypoints=True,coordinates=coordinates, **config());