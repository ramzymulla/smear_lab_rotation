import cv2
import numpy as np

# File paths
full_vid_path = '/Volumes/CrucialRZA/smearlab/clickbait-loco/thermister/2209/t7/04182026_2209_t7.avi'
bg_output_path = '/Volumes/CrucialRZA/smearlab/clickbait-loco/thermister/2209/t7/04182026_2209_t7_static_background.png'
num_sample_frames = 50

cap = cv2.VideoCapture(full_vid_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Generate random frame indices
frame_ids = np.random.uniform(0, total_frames, size=num_sample_frames).astype(int)

frames = []
for fid in frame_ids:
    cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
    ret, frame = cap.read()
    if ret:
        frames.append(frame)

cap.release()

# Calculate median to remove moving objects
median_frame = np.nanmedian(frames, axis=0).astype(dtype=np.uint8)
cv2.imwrite(bg_output_path, median_frame)