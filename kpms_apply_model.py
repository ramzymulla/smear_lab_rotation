import keypoint_moseq as kpms
import matplotlib.pyplot as plt
import os
import re
import fnmatch
import sys
import numpy as np
from pathlib import Path
import jax_moseq


framerate = sys.argv[1]
sleap_dir = sys.argv[2]
includeStr = sys.argv[3].split('/')[-1]


project_dir = str(Path(f"/home/rza/Research/smearlab/scripts/kpms_v3/"))
config = lambda: kpms.load_config(project_dir)
base_path_str = "/home/rza/Research/smearlab/clickbait-loco/thermister/"
model_name = f'kpms_model_{framerate}fps_v3'

sleap_file = f"{sleap_dir}/*{includeStr}.slp"  # any .slp or .h5 file with predictions for a single video
video_dir = str(Path(base_path_str))  # any directory containing videos (e.g. .mp4, .avi) and/or sleap files with predictions for those videos
if not os.path.exists(project_dir):
    kpms.setup_project(project_dir, sleap_file=sleap_file,video_dir=video_dir) 

kpms.update_config(project_dir,
                    use_bodyparts=['nose1','neck1','centroid1','forepawL1','forepawR1','hindpawL1','hindpawR1','tailstart1'],
                    anterior_bodyparts=["nose1","neck1"],posterior_bodyparts=["tailstart1"],fps=int(framerate),
                    video_dir=video_dir)
model = kpms.load_checkpoint(project_dir, model_name)[0]

sleap_files = [str(i) for i in list(Path(sleap_dir).glob(includeStr.replace('.avi','.slp')))]
print(sleap_files)

for subject in ['2208','2209','2210']:
# load data 
    if subject not in includeStr:
        continue
    for sleap_file in sleap_files:
        keypoint_data_path = sleap_file # can be a file, a directory, or a list of files
        
        outDir = os.path.join(project_dir,subject)
        if not os.path.exists(outDir):
            os.mkdir(outDir)

        try:
            coordinates, confidences, bodyparts = kpms.load_keypoints(keypoint_data_path, "sleap")
            coordinates, confidences = kpms.outlier_removal(
                coordinates,
                confidences,
                project_dir,
                overwrite=False,
                **config()
            )
        except:
            continue

        data, metadata = kpms.format_data(coordinates, confidences, **config())
        data = jax_moseq.utils.debugging.convert_data_precision(data)

        

        # apply saved model to new data
        results = kpms.apply_model(model, data, metadata, project_dir, model_name, overwrite=True, **config())

        # results = kpms.load_results(project_dir, model_name)

        kpms.save_results_as_csv(results, outDir, model_name)

        kpms.generate_trajectory_plots(coordinates, results, outDir, model_name, **config())

        kpms.plot_similarity_dendrogram(coordinates, results, outDir, model_name, **config())

        kpms.generate_grid_movies(results, outDir, model_name, 
                                overlay_keypoints=True,coordinates=coordinates, **config());
