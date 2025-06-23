import argparse
import time
from glob import glob

import cv2
import numpy as np
from tqdm import tqdm
from utils import drawTrajectory


def rot2quat(R):
    from scipy.spatial.transform import Rotation
    r = Rotation.from_matrix(R)
    q = r.as_quat()  # [qx, qy, qz, qw]
    return q[3], q[0], q[1], q[2]  # Reorder to qw, qx, qy, qz


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir_root", type=str, default="../data/",
                    help="dataset root")
parser.add_argument("--dataset_type", type=str, default='KITTI', choices=['KITTI', 'TUM'],
                    help="dataset type")
parser.add_argument("--len_trajMap", type=int, default=700,
                    help="size of the trajectory map")
args = parser.parse_args()

if __name__ == "__main__":
    # Define dataset and intrinsic parameters
    seq = '09'
    img_data_dir = args.data_dir_root + 'kitti-odom/' + seq + '/image_0/'
    dataset_reader = DatasetReaderKITTI(args.data_dir_root + 'kitti-odom/' + seq)
    width = 1241.0
    height = 376.0
    fx, fy, cx, cy = [718.8560, 718.8560, 607.1928, 185.2157]

    # Initialize output pose file and trajectory map
    out_pose_file = f'KITTI-{seq}-traj_est.txt'
    trajMap = np.zeros((args.len_trajMap, args.len_trajMap, 3), dtype=np.uint8)

    # Get image list
    img_list = glob(img_data_dir + '/*.png')
    img_list.sort()
    num_frames = len(img_list)

    # Initialize ORB and matcher
    orb = cv2.ORB_create(nfeatures=6000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Initialize pose and trajectory lists
    curr_R = np.eye(3)
    curr_t = np.array([0, 0, 0], dtype=np.float64)
    prev_R = np.eye(3)
    prev_t = np.array([0, 0, 0], dtype=np.float64)
    track_positions = []
    kitti_positions = []
    processing_times = []

    # Process frames with progress bar
    for i in tqdm(range(num_frames), desc="Processing frames", unit="frame"):
        start_time = time.time()

        # Load image
        curr_img = cv2.imread(img_list[i], 0)
        if i == 0:
            prev_img = curr_img
            continue

        # ORB feature detection and matching
        kp1, des1 = orb.detectAndCompute(prev_img, None)
        kp2, des2 = orb.detectAndCompute(curr_img, None)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
        print(f"{len(pts2)} features left after feature matching.")

        # Compute essential matrix
        E, mask = cv2.findEssentialMat(pts1, pts2, focal=fx, pp=(cx, cy), method=cv2.RANSAC, prob=0.999, threshold=1)
        pts1 = pts1[mask.ravel() == 1]
        pts2 = pts2[mask.ravel() == 1]
        print(f"{len(pts2)} features left after pose estimation.")

        # Recover pose
        _, R, t, mask = cv2.recoverPose(E, pts1, pts2, focal=fx, pp=(cx, cy))

        # Get ground truth position and scale
        kitti_pos, kitti_scale = dataset_reader.readGroundtuthPosition(i)
        kitti_pos = np.asarray(kitti_pos).flatten()
        if kitti_pos.shape != (3,):
            print(f"[ERROR] kitti_pos has incorrect shape: {kitti_pos.shape}")
            continue
        # if kitti_scale <= 0.1:
        #     print(f"[WARNING] Skipping frame {i}: kitti_scale ({kitti_scale}) <= 0.1")
        #     continue

        # Compute camera motion
        R = R.transpose()
        t = -np.matmul(R, t).flatten() * kitti_scale  # Apply scale
        if i == 1:
            curr_R = R
            curr_t = t
        else:
            curr_R = np.matmul(prev_R, R)
            curr_t = np.matmul(prev_R, t) + prev_t

        # Append positions
        track_positions.append(curr_t)
        kitti_positions.append(kitti_pos)

        # Save pose to file
        end_time = time.time()
        timestamp = end_time - start_time
        processing_times.append(timestamp)
        qw, qx, qy, qz = rot2quat(curr_R)
        with open(out_pose_file, 'a') as f:
            f.write('%f %f %f %f %f %f %f %f\n' % (
                float(timestamp), float(curr_t[0]), float(curr_t[1]), float(curr_t[2]),
                float(qx), float(qy), float(qz), float(qw)
            ))

        # Draw trajectories
        if len(track_positions) >= 2 and len(kitti_positions) >= 2:
            track_positions_array = np.array(track_positions, dtype=np.float64)
            kitti_positions_array = np.array(kitti_positions, dtype=np.float64)
            drawTrajectory(trajMap, track_positions_array, kitti_positions_array, i, args.len_trajMap)
        else:
            print(f"[INFO] Skipping trajectory update for frame {i}: need at least 2 points")

        # Draw keypoints and matches
        img_matching = cv2.drawMatches(prev_img, kp1, curr_img, kp2, matches[:100], None)
        height, width = img_matching.shape[:2]
        img_matching = cv2.resize(img_matching, (int(width / 2), int(height / 2)))
        cv2.imshow('feature matching', img_matching)
        curr_img_kp = cv2.drawKeypoints(curr_img, kp2, None, color=(0, 255, 0), flags=0)
        cv2.imshow('keypoints from current image', curr_img_kp)

        if cv2.waitKey(1) == ord('q'):
            break

        prev_img = curr_img
        prev_R = curr_R
        prev_t = curr_t

    # Calculate and display average FPS
    avg_processing_time = np.mean(processing_times)
    avg_fps = 1.0 / avg_processing_time if avg_processing_time > 0 else 0
    print(f"Average processing time per frame: {avg_processing_time:.4f} seconds")
    print(f"Average FPS: {avg_fps:.2f}")

    # Save trajectory image
    cv2.imwrite(f'KITTI-{seq}_trajMap.png', trajMap)
    cv2.destroyAllWindows()
