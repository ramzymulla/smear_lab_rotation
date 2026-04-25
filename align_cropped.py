import cv2

# File paths
bg_img_path = '/Volumes/CrucialRZA/smearlab/clickbait-loco/thermister/2209/t7/04182026_2209_t7_static_background.png'
crop_vid_path = '/Volumes/CrucialRZA/smearlab/clickbait-loco/thermister/2209/t7/04182026_2209_t7_raw.avi'
output_vid_path = '/Volumes/CrucialRZA/smearlab/clickbait-loco/thermister/2209/t7/04182026_2209_t7_raw_aligned.avi'

# Load the static background
bg_img = cv2.imread(bg_img_path)
bg_gray = cv2.cvtColor(bg_img, cv2.COLOR_BGR2GRAY)
height, width = bg_img.shape[:2]

cap_crop = cv2.VideoCapture(crop_vid_path)
fps = cap_crop.get(cv2.CAP_PROP_FPS)

# Initialize video writer
fourcc = cv2.VideoWriter_fourcc(*'h264')
out = cv2.VideoWriter(output_vid_path, fourcc, fps, (width, height))

while True:
    ret_crop, frame_crop = cap_crop.read()

    if not ret_crop:
        break

    crop_gray = cv2.cvtColor(frame_crop, cv2.COLOR_BGR2GRAY)

    # Perform template matching
    res = cv2.matchTemplate(bg_gray, crop_gray, cv2.TM_CCOEFF_NORMED)
    _, _, _, max_loc = cv2.minMaxLoc(res)

    x, y = max_loc
    h, w = frame_crop.shape[:2]

    # Copy the clean background and overlay the crop
    out_frame = bg_img.copy()
    out_frame[y:y+h, x:x+w] = frame_crop

    out.write(out_frame)

cap_crop.release()
out.release()