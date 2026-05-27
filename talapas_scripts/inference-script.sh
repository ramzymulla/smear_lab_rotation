#!/bin/bash
sleap track --gui --data_path /Volumes/CrucialRZA/smearlab/labels_v002.pkg.slp --video_index 0 --video_dataset video0/video --video_input_format channels_last --frames 66560 --model_paths models/260414_140714.single_instance.n=157 -o labels_v002.pkg.slp.predictions.slp
