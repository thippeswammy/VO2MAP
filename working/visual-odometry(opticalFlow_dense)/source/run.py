import argparse
import json
import os
import subprocess
import time
from pprint import pprint

import cv2
import numpy as np
import plotly.graph_objects as go
from tqdm import tqdm

from feature_tracking import FeatureTracker
from kitti_reader import DatasetReaderKITTI
from utils import drawFrameFeatures, drawTrajectory


def plot_3d_trajectories_interactive(estimated, groundtruth, title="3D Trajectory Comparison"):
    estimated = np.asarray(estimated, dtype=np.float32)
    groundtruth = np.asarray(groundtruth, dtype=np.float32)

    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=estimated[:, 0], y=estimated[:, 1], z=estimated[:, 2],
        mode='lines+markers',
        marker=dict(size=3, color='blue'),
        line=dict(color='blue'),
        name='Estimated'
    ))

    fig.add_trace(go.Scatter3d(
        x=groundtruth[:, 0], y=groundtruth[:, 1], z=groundtruth[:, 2],
        mode='lines+markers',
        marker=dict(size=3, color='green'),
        line=dict(color='green', dash='dash'),
        name='Groundtruth'
    ))

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X (meters)',
            yaxis_title='Y (meters)',
            zaxis_title='Z (meters)'
        ),
        legend=dict(x=0.02, y=0.98)
    )

    fig.write_html("trajectory_3d_interactive.html")
    fig.show()


def rot2quat(R):
    from scipy.spatial.transform import Rotation
    r = Rotation.from_matrix(R)
    q = r.as_quat()  # [qx, qy, qz, qw]
    return q[3], q[0], q[1], q[2]  # Reorder to qw, qx, qy, qz


def compute_absolute_trajectory_error(estimated, groundtruth):
    estimated = np.asarray(estimated)
    groundtruth = np.asarray(groundtruth)

    if estimated.shape != groundtruth.shape:
        raise ValueError(f"Shape mismatch: estimated {estimated.shape}, groundtruth {groundtruth.shape}")

    errors = np.linalg.norm(estimated - groundtruth, axis=1)
    ate = np.sqrt(np.mean(errors ** 2))
    return ate


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir_root", type=str, default="../../../data/", help="dataset root")
parser.add_argument("--dataset_type", type=str, default='KITTI', choices=['KITTI', 'TUM'], help="dataset type")
parser.add_argument("--len_trajMap", type=int, default=20000, help="size of the trajectory map")
args = parser.parse_args()

if __name__ == "__main__":
    PID = os.getpid()
    stop_file = "stop.txt"
    output_file = "usage.json"

    # Remove stop file if it exists
    if os.path.exists(stop_file):
        os.remove(stop_file)

    monitor_cmd = ["python", r"F:\RunningProjects\VisualOdemetry\Visual-odometry-tutorial\utils\monitor.py", "--pid",
                   str(PID), "--output_file", output_file, "--stop_file", stop_file]
    monitor_process = subprocess.Popen(monitor_cmd)
    print(f"Started monitoring process with PID {monitor_process.pid}")

    Start_time = time.time_ns()

    seq = '09'
    dataset_reader = DatasetReaderKITTI(args.data_dir_root + 'kitti-odom/' + seq)
    K = dataset_reader.readCameraMatrix()

    out_pose_file = f'KITTI-{seq}-traj_est.txt'
    trajMap = np.zeros((args.len_trajMap, args.len_trajMap, 3), dtype=np.uint8)

    tracker = FeatureTracker()
    detector = cv2.GFTTDetector_create(maxCorners=6000, qualityLevel=0.01, minDistance=1)

    prev_points = np.empty(0)
    prev_frame_BGR = dataset_reader.readFrame(0)
    kitti_positions, track_positions = [], []
    camera_rot = np.eye(3)
    camera_pos, _ = dataset_reader.readGroundtuthPosition(0)

    if os.path.exists(out_pose_file):
        os.remove(out_pose_file)
    for frame_no in tqdm(range(1, min(dataset_reader._numFrames, 10000)), desc="Processing frames", unit="frame"):
        start_time = time.time()

        curr_frame_BGR = dataset_reader.readFrame(frame_no)
        prev_img = cv2.cvtColor(prev_frame_BGR, cv2.COLOR_BGR2GRAY)
        curr_img = cv2.cvtColor(curr_frame_BGR, cv2.COLOR_BGR2GRAY)

        prev_points = detector.detect(prev_img)
        prev_points = cv2.KeyPoint_convert(sorted(prev_points, key=lambda p: p.response, reverse=True))

        prev_points, curr_points = tracker.trackFeatures(prev_img, curr_img, prev_points, removeOutliers=True)

        E, mask = cv2.findEssentialMat(curr_points, prev_points, K, cv2.RANSAC, 0.99, 1.0, None)
        prev_points = np.array([pt for (idx, pt) in enumerate(prev_points) if mask[idx] == 1])
        curr_points = np.array([pt for (idx, pt) in enumerate(curr_points) if mask[idx] == 1])
        _, R, T, _ = cv2.recoverPose(E, curr_points, prev_points, K)

        kitti_pos, kitti_scale = dataset_reader.readGroundtuthPosition(frame_no)

        T = np.asarray(T).flatten()
        camera_pos = np.asarray(camera_pos).flatten()
        rotated_T = camera_rot.dot(T)
        camera_pos = camera_pos + kitti_scale * rotated_T
        camera_rot = R.dot(camera_rot)

        kitti_pos = np.asarray(kitti_pos).flatten()
        if kitti_pos.shape != (3,):
            print(f"[ERROR] kitti_pos has incorrect shape: {kitti_pos.shape}")
            continue

        kitti_positions.append(kitti_pos)
        track_positions.append(camera_pos)

        transformation_matrix = np.hstack((camera_rot, camera_pos.reshape(3, 1)))
        pose_line = ' '.join(map(str, transformation_matrix.flatten()))

        with open(out_pose_file, 'a') as f:
            f.write(f"{pose_line}\n")

        if len(track_positions) >= 2 and len(kitti_positions) >= 2:
            track_positions_array = np.array(track_positions, dtype=np.float64)
            kitti_positions_array = np.array(kitti_positions, dtype=np.float64)
            drawTrajectory(trajMap, track_positions_array, kitti_positions_array, frame_no, args.len_trajMap)

        drawFrameFeatures(curr_img, prev_points, curr_points, frame_no)

        if cv2.waitKey(1) == ord('q'):
            break

        prev_points, prev_frame_BGR = curr_points, curr_frame_BGR

    End_time = time.time_ns()
    total_time = (End_time - Start_time) / 1e9
    num_frames = dataset_reader._numFrames
    fps = num_frames / total_time if total_time > 0 else 0

    print(f"Total Time = {total_time:.2f} s")
    print(f"Frames processed = {num_frames}")
    print(f"Average FPS (by processing time) = {fps:.2f} fps")
    # Signal monitoring process to stop
    with open(stop_file, 'w') as f:
        f.write("stop")

    # Wait for monitoring process to finish
    monitor_process.wait()

    # Read and display monitoring results
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            avg_result = json.load(f)
        print("\n[AVERAGE USAGE]")
        pprint(avg_result)
        os.remove(output_file)  # Clean up
    else:
        print("No monitoring results found.")

    # Clean up stop file
    if os.path.exists(stop_file):
        os.remove(stop_file)
    if len(kitti_positions) == len(track_positions):
        estimated_aligned = np.copy(track_positions)
        kitti_positions_np = np.asarray(kitti_positions, dtype=np.float32)
        np.save('estimated_aligned.npy', estimated_aligned)
        np.save('kitti_positions.npy', kitti_positions_np)
        plot_3d_trajectories_interactive(estimated_aligned, kitti_positions)
        ate = compute_absolute_trajectory_error(track_positions, kitti_positions)
        print(f"\n\n[RESULT] Absolute Trajectory Error (ATE): {ate:.4f} meters")
    else:
        print(
            f"\n\n[ERROR] Length mismatch: track_positions={len(track_positions)}, kitti_positions={len(kitti_positions)}")

    cv2.imwrite(f'KITTI-{seq}_trajMap.png', cv2.flip(trajMap, 0))
    cv2.destroyAllWindows()
