import os
import cv2

inp = r"20250908_192321.mp4"       # your input video
out_dir = r"CameraCal/SamsungImg"     # output folder
os.makedirs(out_dir, exist_ok=True)

cap = cv2.VideoCapture(inp)
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
frame_idx = 0
success, frame = cap.read()

while success:
    # timestamp in seconds
    timestamp = frame_idx / fps
    filename = os.path.join(out_dir, f"image_{frame_idx:05d}.png")

    cv2.imwrite(filename, frame)

    # set file modification time = video timestamp
    ts_unix = os.path.getmtime(inp) - cap.get(cv2.CAP_PROP_FRAME_COUNT)/fps + timestamp
    os.utime(filename, (ts_unix, ts_unix))

    frame_idx += 1
    success, frame = cap.read()

cap.release()
print(f"Saved {frame_idx} frames to {out_dir}")
