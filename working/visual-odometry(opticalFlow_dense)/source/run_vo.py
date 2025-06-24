import argparse
import os
import subprocess
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from kitti_reader import DatasetReaderKITTI


def rot2quat(R):
    from scipy.spatial.transform import Rotation
    r = Rotation.from_matrix(R)
    q = r.as_quat()  # [qx, qy, qz, qw]
    return q[3], q[0], q[1], q[2]  # Reorder to qw, qx, qy, qz


def plot_3d_trajectories(estimated, groundtruth, title="3D Trajectory Comparison"):
    estimated = np.asarray(estimated, dtype=np.float32)
    groundtruth = np.asarray(groundtruth, dtype=np.float32)

    if estimated.shape[1] != 3 or groundtruth.shape[1] != 3:
        raise ValueError(f"Expected 3D coordinates. Got shapes {estimated.shape}, {groundtruth.shape}")

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    # Mark starting points
    ax.scatter(estimated[0, 0], estimated[0, 1], estimated[0, 2], color='blue', s=50, marker='o',
               label='Start Estimated')
    ax.scatter(groundtruth[0, 0], groundtruth[0, 1], groundtruth[0, 2], color='green', s=50, marker='x',
               label='Start Groundtruth')

    ax.plot(estimated[:, 0], estimated[:, 1], estimated[:, 2], 'b-', label='Estimated', linewidth=2)
    ax.plot(groundtruth[:, 0], groundtruth[:, 1], groundtruth[:, 2], 'g--', label='Groundtruth', linewidth=2)

    ax.set_title(title)
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_zlabel('Z (meters)')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig("trajectory_plot.png", dpi=300)
    plt.show()


def compute_absolute_trajectory_error(estimated, groundtruth):
    estimated = np.asarray(estimated)
    groundtruth = np.asarray(groundtruth)

    if estimated.shape != groundtruth.shape:
        raise ValueError(f"Shape mismatch: estimated {estimated.shape}, groundtruth {groundtruth.shape}")

    errors = np.linalg.norm(estimated - groundtruth, axis=1)
    ate = np.sqrt(np.mean(errors ** 2))
    return ate


def drawTrajectory_vo(trajMap, trackedPoints, groundtruthPoints, frame_no, len_trajMap):
    offset_draw = int(len_trajMap / 2)
    offset_draw_small = int(len_trajMap / 8)

    # Draw estimated trajectory (blue)
    x_est = int(trackedPoints[-1, 0] + offset_draw)
    z_est = int(-trackedPoints[-1, 2] + offset_draw_small)
    if 0 <= x_est < len_trajMap and 0 <= z_est < len_trajMap:
        cv2.circle(trajMap, (x_est, z_est), 1, (255, 0, 0), 5)

    # Draw ground truth trajectory (white)
    x_gt = int(groundtruthPoints[-1, 0] + offset_draw)
    z_gt = int(groundtruthPoints[-1, 2] + offset_draw_small)
    if 0 <= x_gt < len_trajMap and 0 <= z_gt < len_trajMap:
        cv2.circle(trajMap, (x_gt, z_gt), 1, (255, 255, 255), 2)

    trajMap1 = cv2.flip(trajMap, 0)
    cv2.imshow('Trajectory', cv2.resize(trajMap1, (1000, 700)))


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir_root", type=str, default="../../../data/", help="dataset root")
parser.add_argument("--dataset_type", type=str, default='KITTI', choices=['KITTI', 'TUM'], help="dataset type")
parser.add_argument("--len_trajMap", type=int, default=2000, help="size of the trajectory map")
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
    width = 1241.0
    height = 376.0
    fx, fy, cx, cy = [718.8560, 718.8560, 607.1928, 185.2157]

    seq = '09'
    dataset_reader = DatasetReaderKITTI(args.data_dir_root + 'kitti-odom/' + seq)
    K = dataset_reader.readCameraMatrix()

    out_pose_file = f'KITTI-{seq}-traj_est_by_vo.txt'
    trajMap = np.zeros((args.len_trajMap, args.len_trajMap, 3), dtype=np.uint8)
    if os.path.exists(out_pose_file):
        os.remove(out_pose_file)
    orb = cv2.ORB_create(nfeatures=6000)

    prev_frame_BGR = dataset_reader.readFrame(0)
    kitti_positions, track_positions = [], []
    camera_pos, _, camera_rot, _ = dataset_reader.readGroundTruthPositionRotation(0)

    prev_R = np.eye(3)
    prev_t = np.zeros((3, 1))
    processing_times = []

    for frame_no in tqdm(range(1, min(dataset_reader._numFrames, 2000)), desc="Processing frames", unit="frame"):
        start_time = time.time()

        curr_frame_BGR = dataset_reader.readFrame(frame_no)
        prev_img = cv2.cvtColor(prev_frame_BGR, cv2.COLOR_BGR2GRAY)
        curr_img = cv2.cvtColor(curr_frame_BGR, cv2.COLOR_BGR2GRAY)

        kp1, des1 = orb.detectAndCompute(prev_img, None)
        kp2, des2 = orb.detectAndCompute(curr_img, None)

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)

        if len(matches) < 8:
            print(f"[WARNING] Not enough matches in frame {frame_no}")
            continue

        img_matching = cv2.drawMatches(prev_img, kp1, curr_img, kp2, matches[:100], None)
        h, w, _ = img_matching.shape
        img_matching = cv2.resize(img_matching, (w // 2, h // 2))
        cv2.imshow('feature matching', img_matching)
        cv2.waitKey(1)

        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

        E, mask = cv2.findEssentialMat(pts1, pts2, focal=fx, pp=(cx, cy), method=cv2.RANSAC, prob=0.999, threshold=1.0)
        if E is None or mask is None:
            print(f"[WARNING] Essential matrix could not be computed for frame {frame_no}")
            continue

        pts1 = pts1[mask.ravel() == 1]
        pts2 = pts2[mask.ravel() == 1]

        if len(pts1) < 8:
            print(f"[WARNING] Not enough inliers after essential matrix filtering at frame {frame_no}")
            continue

        _, R, t, _ = cv2.recoverPose(E, pts1, pts2, focal=fx, pp=(cx, cy))

        curr_R = prev_R @ R
        curr_t = prev_t + prev_R @ t

        T = t  # t is the relative translation
        camera_rot = camera_rot @ R
        camera_pos = camera_pos + (camera_rot @ T).flatten()

        kitti_pos, kitti_scale = dataset_reader.readGroundtuthPosition(frame_no)
        # if kitti_scale <= 0.1:
        #     print(f"[WARNING] Skipping frame {frame_no}: scale too small ({kitti_scale})")
        #     continue
        kitti_positions.append(np.asarray(kitti_pos).flatten())
        track_positions.append(np.asarray(camera_pos).flatten())

        end_time = time.time()
        timestamp = end_time - start_time
        processing_times.append(timestamp)

        # qw, qx, qy, qz = rot2quat(camera_rot)
        # with open(out_pose_file, 'a') as f:
        #     f.write('%f %f %f %f %f %f %f %f\n' % (
        #         float(timestamp), float(camera_pos[0]), float(camera_pos[1]), float(-camera_pos[2]),
        #         float(qx), float(qy), float(qz), float(qw)
        #     ))
        transformation_matrix = np.hstack((camera_rot, camera_pos.reshape(3, 1)))  # shape: (3, 4)
        pose_line = ' '.join(map(str, transformation_matrix.flatten()))  # row-major flatten
        with open(out_pose_file, 'a') as f:
            f.write(f"{pose_line}\n")

        if len(track_positions) >= 2 and len(kitti_positions) >= 2:
            drawTrajectory_vo(trajMap, np.array(track_positions), np.array(kitti_positions), frame_no, args.len_trajMap)

        if cv2.waitKey(1) == ord('q'):
            break

        prev_frame_BGR = curr_frame_BGR
        prev_R = curr_R
        prev_t = curr_t

    avg_processing_time = np.mean(processing_times)
    avg_fps = 1.0 / avg_processing_time if avg_processing_time > 0 else 0
    End_time = time.time_ns()
    total_time = (End_time - Start_time) / 1e9
    num_frames = dataset_reader._numFrames
    fps = num_frames / total_time if total_time > 0 else 0
    print(f"Total Time = {total_time:.2f} s")
    print(f"Frames processed = {num_frames}")
    # print(f"Average Processing Time per Frame = {avg_processing_time:.4f} s")
    print(f"FPS = {avg_fps:.2f} fps")
    # print(f"Actual FPS (by total time) = {fps:.2f} fps")
    # print(f"Average FPS (by processing time) = {fps:.2f} fps")
    # Read and display monitoring results
    # Signal monitoring process to stop

    if len(kitti_positions) == len(track_positions):
        estimated_aligned = np.copy(track_positions)
        estimated_aligned[:, 2] = -estimated_aligned[:, 2]
        kitti_positions_np = np.asarray(kitti_positions, dtype=np.float32)
        plot_3d_trajectories(estimated_aligned, kitti_positions)
        ate = compute_absolute_trajectory_error(estimated_aligned, kitti_positions)
        print(f"\n\n[RESULT] Absolute Trajectory Error (ATE): {ate:.4f} meters")
    else:
        print(f"\n\n[ERROR] Length mismatch between trajectory and groundtruth")

    cv2.imwrite(f'KITTI-{seq}_trajMap.png', cv2.flip(trajMap, 0))
    cv2.destroyAllWindows()
