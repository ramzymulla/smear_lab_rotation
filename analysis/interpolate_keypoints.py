import keypoint_moseq as kpms
import matplotlib.pyplot as plt
import os
from pathlib import Path
import numpy as np
import sleap_io as sio


base_path_str = "/Volumes/CrucialRZA/smearlab/clickbait-loco/thermister/"
rawModel = "labels_v007/models"
regModel = "labels_v003_0.5scaling/models"
sleap_dir = str(Path(f"{base_path_str}/"))

regFnames = [str(i) for i in list(Path(base_path_str+regModel).glob("**/preds/*t[1-7].slp"))]
rawFnames = [str(i) for i in list(Path(base_path_str+rawModel).glob("**/preds/*t[1-7]_raw.slp"))]
outDir = os.path.join(base_path_str,"interpolated_keypoints")
if not os.path.exists(outDir):
    os.mkdir(outDir)

for sleap_file in regFnames + rawFnames:
    outFile = os.path.join(outDir,os.path.basename(sleap_file).replace('.slp','_interpolated.csv'))

    if "interpol" in sleap_file or os.path.basename(sleap_file)[:2] == "._" or os.path.exists(outFile):
        continue

    project_dir = os.path.join(os.getcwd(),"tmp")
    kpms.setup_project(project_dir, sleap_file=sleap_file, overwrite=True)

    config = lambda: kpms.load_config(project_dir)

    # load data 
    keypoint_data_path = sleap_file  # can be a file, a directory, or a list of files
    coordinates, confidences, bodyparts = kpms.load_keypoints(keypoint_data_path, "sleap")


    kpms.update_config(project_dir,
                    anterior_bodyparts=["nose1","neck1"],posterior_bodyparts=["tailstart1"],
                    outlier_scale_factor=6)

    
    
    

    # bad_mask = (confidences < 0.5)

    coordinates, confidences = kpms.outlier_removal(
                                                    coordinates,
                                                    confidences,
                                                    project_dir,
                                                    overwrite=False,
                                                    **config()
                                                )
    
    if isinstance(coordinates,dict):
        curr_key = list(coordinates.keys())[0]
        coordinates = coordinates[curr_key]

    if isinstance(confidences,dict):
        curr_key = list(confidences.keys())[0]
        confidences = confidences[curr_key]

    combinedArray = np.concatenate((coordinates,confidences[:,:,None]),axis=-1)

    labels = sio.load_file(sleap_file)
    labels.update_from_numpy(combinedArray[:,None,:,:],tracks=[sio.Track('interpolated')])
    sio.save_csv(labels,outFile)
    labels.save(os.path.join(outDir,os.path.basename(sleap_file).replace('.slp','_interpolated.slp')))




