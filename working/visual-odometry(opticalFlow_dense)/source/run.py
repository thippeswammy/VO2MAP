import argparse
import time

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

    # Estimated trajectory
    fig.add_trace(go.Scatter3d(
        x=estimated[:, 0], y=estimated[:, 1], z=estimated[:, 2],
        mode='lines+markers',
        marker=dict(size=3, color='blue'),
        line=dict(color='blue'),
        name='Estimated'
    ))

    # Groundtruth trajectory
    fig.add_trace(go.Scatter3d(
        x=groundtruth[:, 0], y=groundtruth[:, 1], z=groundtruth[:, 2],
        mode='lines+markers',
        marker=dict(size=3, color='green'),
        line=dict(color='green', dash='dash'),
        name='Groundtruth'
    ))

    # Layout
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
    """
    Compute Absolute Trajectory Error (ATE)
    Params:
        estimated: Nx3 array of estimated positions
        groundtruth: Nx3 array of groundtruth positions
    Returns:
        ate: float, root mean squared error
    """
    estimated = np.asarray(estimated)
    groundtruth = np.asarray(groundtruth)

    if estimated.shape != groundtruth.shape:
        raise ValueError(f"Shape mismatch: estimated {estimated.shape}, groundtruth {groundtruth.shape}")

    errors = np.linalg.norm(estimated - groundtruth, axis=1)
    ate = np.sqrt(np.mean(errors ** 2))
    return ate


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir_root", type=str,
                    default="../../../data/",
                    help="dataset root")
parser.add_argument("--dataset_type", type=str, default='KITTI', choices=['KITTI', 'TUM'],
                    help="dataset type")
parser.add_argument("--len_trajMap", type=int, default=2000,
                    help="size of the trajectory map")
args = parser.parse_args()

if __name__ == "__main__":
    # Initialize dataset and camera parameters
    seq = '09'
    dataset_reader = DatasetReaderKITTI(args.data_dir_root + 'kitti-odom/' + seq)
    K = dataset_reader.readCameraMatrix()

    # Initialize output pose file and trajectory map
    out_pose_file = f'KITTI-{seq}-traj_est.txt'
    trajMap = np.zeros((args.len_trajMap, args.len_trajMap, 3), dtype=np.uint8)

    # Initialize feature detector and tracker
    tracker = FeatureTracker()
    detector = cv2.GFTTDetector_create(maxCorners=6000, qualityLevel=0.01, minDistance=1)

    # Initialize pose and points
    prev_points = np.empty(0)
    prev_frame_BGR = dataset_reader.readFrame(0)
    kitti_positions, track_positions = [], []
    camera_rot = np.eye(3)  # Identity rotation matrix
    camera_pos, _ = dataset_reader.readGroundtuthPosition(0)

    processing_times = []

    # Process frames
    for frame_no in tqdm(range(1, min(dataset_reader._numFrames, 10000)), desc="Processing frames", unit="frame"):
        start_time = time.time()

        # Load and convert frames
        curr_frame_BGR = dataset_reader.readFrame(frame_no)
        prev_img = cv2.cvtColor(prev_frame_BGR, cv2.COLOR_BGR2GRAY)
        curr_img = cv2.cvtColor(curr_frame_BGR, cv2.COLOR_BGR2GRAY)

        # Feature detection
        prev_points = detector.detect(prev_img)
        prev_points = cv2.KeyPoint_convert(sorted(prev_points, key=lambda p: p.response, reverse=True))

        # Feature tracking (optical flow)
        prev_points, curr_points = tracker.trackFeatures(prev_img, curr_img, prev_points, removeOutliers=True)
        # print(f"{len(curr_points)} features left after feature tracking.")

        # Essential matrix and pose estimation
        E, mask = cv2.findEssentialMat(curr_points, prev_points, K, cv2.RANSAC, 0.99, 1.0, None)
        prev_points = np.array([pt for (idx, pt) in enumerate(prev_points) if mask[idx] == 1])
        curr_points = np.array([pt for (idx, pt) in enumerate(curr_points) if mask[idx] == 1])
        _, R, T, _ = cv2.recoverPose(E, curr_points, prev_points, K)
        # print(f"{len(curr_points)} features left after pose estimation.")

        # Read ground truth and scale
        kitti_pos, kitti_scale = dataset_reader.readGroundtuthPosition(frame_no)
        if kitti_scale <= 0.1:
            print(f"[WARNING] Skipping frame {frame_no}: kitti_scale ({kitti_scale}) <= 0.1")
            continue

        # Update pose
        T = np.asarray(T).flatten()
        camera_pos = np.asarray(camera_pos).flatten()
        rotated_T = camera_rot.dot(T)
        camera_pos = camera_pos + kitti_scale * rotated_T
        camera_rot = R.dot(camera_rot)

        # Debug shapes
        # print(f"Frame {frame_no}:")
        # print(f"T shape: {T.shape}, rotated_T shape: {rotated_T.shape}, camera_pos shape: {camera_pos.shape}")
        if T.shape != (3,) or rotated_T.shape != (3,):
            print(f"[ERROR] Shape mismatch: T={T.shape}, rotated_T={rotated_T.shape}, camera_pos={camera_pos.shape}")
            continue

        # Convert kitti_pos
        kitti_pos = np.asarray(kitti_pos).flatten()
        if kitti_pos.shape != (3,):
            print(f"[ERROR] kitti_pos has incorrect shape: {kitti_pos.shape}")
            continue
        # print(f"kitti_pos: {kitti_pos}, shape: {kitti_pos.shape}")

        kitti_positions.append(kitti_pos)
        track_positions.append(camera_pos)

        # Save pose to file
        end_time = time.time()
        timestamp = end_time - start_time
        processing_times.append(timestamp)
        qw, qx, qy, qz = rot2quat(camera_rot)
        with open(out_pose_file, 'a') as f:
            f.write('%f %f %f %f %f %f %f %f\n' % (
                float(timestamp), float(camera_pos[0]), float(camera_pos[1]), float(camera_pos[2]),
                float(qx), float(qy), float(qz), float(qw)
            ))

        # Plot trajectory
        if len(track_positions) >= 2 and len(kitti_positions) >= 2:
            track_positions_array = np.array(track_positions, dtype=np.float64)
            kitti_positions_array = np.array(kitti_positions, dtype=np.float64)
            drawTrajectory(trajMap, track_positions_array, kitti_positions_array, frame_no, args.len_trajMap)
        else:
            print(f"[INFO] Skipping trajectory update for frame {frame_no}: need at least 2 points")

        # Draw keypoints
        drawFrameFeatures(curr_img, prev_points, curr_points, frame_no)

        if cv2.waitKey(1) == ord('q'):
            break

        prev_points, prev_frame_BGR = curr_points, curr_frame_BGR

    # Calculate and display FPS
    # avg_processing_time = np.mean(processing_times)
    # avg_fps = 1.0 / avg_processing_time if avg_processing_time > 0 else 0
    # print(f"Average processing time per frame: {avg_processing_time:.4f} seconds")
    # print(f"Average FPS: {avg_fps:.2f}")

    # Calculate and display FPS
    avg_processing_time = np.mean(processing_times)
    avg_fps = 1.0 / avg_processing_time if avg_processing_time > 0 else 0
    # print(f"Average processing time per frame: {avg_processing_time:.4f} seconds")
    # print(f"Average FPS: {avg_fps:.2f}")

    # Compute ATE
    if len(kitti_positions) == len(track_positions):

        estimated_aligned = np.copy(track_positions)
        # estimated_aligned[:, 0] = 0
        kitti_positions_np = np.asarray(kitti_positions, dtype=np.float32)
        # kitti_positions_np[:, 0] = 0
        np.save('estimated_aligned.npy', estimated_aligned)
        np.save('kitti_positions.npy', kitti_positions_np)
        plot_3d_trajectories_interactive(estimated_aligned, kitti_positions)
        ate = compute_absolute_trajectory_error(track_positions, kitti_positions)
        print(f"[RESULT] Absolute Trajectory Error (ATE): {ate:.4f} meters")
    else:
        print(
            f"[ERROR] Length mismatch: track_positions={len(track_positions)}, kitti_positions={len(kitti_positions)}")

    # Save trajectory image
    cv2.imwrite(f'KITTI-{seq}_trajMap.png', cv2.flip(trajMap, 0))
    cv2.destroyAllWindows()
