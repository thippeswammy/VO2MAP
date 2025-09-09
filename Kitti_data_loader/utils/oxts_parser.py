import datetime
import os


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
            frac = frac[:6]
            time_str = f"{t}.{frac}"

        dt = datetime.datetime.fromisoformat(f"{date_str} {time_str}")
        timestamps.append(dt)

    # Relative times in seconds
    base_time = timestamps[0]
    rel_times = [(ts - base_time).total_seconds() for ts in timestamps]

    return list(zip(lines, timestamps, rel_times))


def parse_oxts_line(line):
    values = line.strip().split()
    vals = list(map(float, values))
    return {
        "lat": vals[0],
        "lon": vals[1],
        "alt": vals[2],
        "roll": vals[3],
        "pitch": vals[4],
        "yaw": vals[5],
        "vn": vals[6],
        "ve": vals[7],
        "vf": vals[8],
        "vl": vals[9],
        "vu": vals[10],
        "ax": vals[11],
        "ay": vals[12],
        "az": vals[13],
        "af": vals[14],
        "al": vals[15],
        "au": vals[16],
        "wx": vals[17],
        "wy": vals[18],
        "wz": vals[19],
        "wf": vals[20],
        "wl": vals[21],
        "wu": vals[22],
        "pos_accuracy": vals[23],
        "vel_accuracy": vals[24],
        "navstat": vals[25],
        "numsats": vals[26],
        "posmode": vals[27],
        "velmode": vals[28],
        "orimode": vals[29],
    }


def load_oxts_data(oxts_dir, timestamp_file):
    timestamp_data = load_timestamps(timestamp_file)
    files = sorted(os.listdir(oxts_dir))
    oxts_data = []

    for (raw_str, ts, rel), fname in zip(timestamp_data, files):
        path = os.path.join(oxts_dir, fname)
        with open(path, "r") as f:
            line = f.readline()
        data = parse_oxts_line(line)
        date_str, time_str = raw_str.split(" ")
        oxts_data.append((date_str, time_str, rel, data))

    return oxts_data
