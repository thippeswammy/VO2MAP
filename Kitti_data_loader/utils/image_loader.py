import datetime
import os

import cv2


def load_timestamps(timestamp_file):
    with open(timestamp_file, "r") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    timestamps = []
    for line in lines:
        # split date and time
        date_str, time_str = line.split(" ")

        # nanoseconds (KITTI has 9 digits, Python only supports 6)
        if "." in time_str:
            t, frac = time_str.split(".")
            frac = frac[:6]  # keep only microseconds
            time_str = f"{t}.{frac}"

        dt = datetime.datetime.fromisoformat(f"{date_str} {time_str}")
        timestamps.append(dt)

    # Relative times in seconds
    base_time = timestamps[0]
    rel_times = [(ts - base_time).total_seconds() for ts in timestamps]

    return list(zip(lines, timestamps, rel_times))


def load_image_list(image_dir, timestamp_file):
    timestamp_data = load_timestamps(timestamp_file)
    files = sorted(os.listdir(image_dir))
    images = []

    for (raw_str, ts, rel), fname in zip(timestamp_data, files):
        path = os.path.join(image_dir, fname)
        img = cv2.imread(path)
        # store as (date, time, rel_sec, image)
        date_str, time_str = raw_str.split(" ")
        images.append((date_str, time_str, rel, img))

    return images
