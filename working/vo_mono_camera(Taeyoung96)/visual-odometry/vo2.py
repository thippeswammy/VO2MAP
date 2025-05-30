import argparse
import os
import time
from glob import glob

import cv2
from tqdm import tqdm

from pose_evaluation_utils import *

print('cuda.getCudaEnabledDeviceCount()', cv2.cuda.getCudaEnabledDeviceCount())

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir_root", type=str, default='D:\downloadFiles\\front_3\VO',
                    help="dataset root")
parser.add_argument("--dataset_type", type=str, default='custom', choices=['KITTI', 'TUM', 'custom'],
                    help="dataset type")
parser.add_argument("--len_trajMap", type=int, default=800,
                    help="size of the trajectory map")
args = parser.parse_args()

# Dataset directory
if args.dataset_type == 'TUM':
    seq = 'freiburg2_desk'
    img_data_dir = os.path.join(args.data_dir_root, 'tum/rgbd_dataset_' + seq + '/rgb/')
elif args.dataset_type == 'KITTI':
    seq = '09'
    img_data_dir = os.path.join(args.data_dir_root, 'kitti-odom/' + seq + '/image_0/')
elif args.dataset_type == 'custom':
    seq = 'custom'
    img_data_dir = os.path.join(args.data_dir_root, 'custom/')
# Intrinsic parameters
if args.dataset_type == 'TUM':
    width = 640
    height = 480
    fx, fy, cx, cy = [1419.95349, 1379.13143, 923.074249, 226.290080]
elif args.dataset_type == 'KITTI':
    # width = 1920.0
    # height = 1080.0
    # fx, fy, cx, cy = [1419.95349, 1379.13143, 923.074249, 226.290080]
    # Dash camera
    width = 1920.0
    height = 1080.0
    fx, fy, cx, cy = [1101.86382, 1114.96528, 885.646472, 443.776090]
elif args.dataset_type == 'custom':
    # width = 1920.0
    # height = 1080.0
    # Zed camera
    # fx, fy, cx, cy = [528.45704204, 528.94853071, 643.5054566, 357.67922371]

    # Dash camera
    # fx, fy, cx, cy = [1419.95349, 1379.13143, 923.074249, 226.290080]
    fx, fy, cx, cy = [1101.86382, 1114.96528, 885.646472, 443.776090]

    width = 1226.0
    height = 370.0
    fx, fy, cx, cy = [718.8560, 718.8560, 607.1928, 185.2157]

if __name__ == "__main__":
    trajMap = np.zeros((args.len_trajMap, args.len_trajMap, 3), dtype=np.uint8)
    out_pose_file = f'./{args.dataset_type}-{seq}-traj_est.txt'
    trajMap1 = trajMap
    if args.dataset_type == 'custom':
        video_files = glob(os.path.join(img_data_dir, '*.mp4'))
        video_files = sorted(video_files)
        if not video_files:
            raise FileNotFoundError("No video file found in custom dataset path.")
        cap = cv2.VideoCapture(video_files[0])
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    else:
        img_list = sorted(glob(os.path.join(img_data_dir, '*.png')))
        num_frames = len(img_list)

    # List to store processing times for each frame
    processing_times = []
    init_tx, init_tz = 0, 0

    # Wrap the loop with tqdm for progress bar
    for i in tqdm(range(num_frames), desc="Processing frames", unit="frame"):
        start_time = time.time()

        if args.dataset_type == 'custom':
            ret, img_cpu = cap.read()
            if not ret:
                break
            img_cpu = cv2.cvtColor(img_cpu, cv2.COLOR_BGR2GRAY)
        else:
            img_cpu = cv2.imread(img_list[i], cv2.IMREAD_GRAYSCALE)

        curr_img = cv2.cuda_GpuMat()
        curr_img.upload(img_cpu)

        if i == 0:
            curr_R = np.eye(3)
            curr_t = np.array([0, 0, 0])
        else:
            if args.dataset_type == 'custom':
                cap.set(cv2.CAP_PROP_POS_FRAMES, i - 1)
                ret, prev_img_cpu = cap.read()
                if not ret:
                    break
                prev_img_cpu = cv2.cvtColor(prev_img_cpu, cv2.COLOR_BGR2GRAY)
            else:
                prev_img_cpu = cv2.imread(img_list[i - 1], cv2.IMREAD_GRAYSCALE)

            prev_img = cv2.cuda_GpuMat()
            prev_img.upload(prev_img_cpu)
            k = cv2.waitKey(10) & 0xff
            if k == 27:
                break
            # ORB GPU
            orb = cv2.cuda_ORB.create(nfeatures=5000)
            kp1_gpu, des1 = orb.detectAndComputeAsync(prev_img, None)
            kp2_gpu, des2 = orb.detectAndComputeAsync(curr_img, None)

            kp1 = orb.convert(kp1_gpu)
            kp2 = orb.convert(kp2_gpu)

            # BFMatcher (GPU)
            bf = cv2.cuda.DescriptorMatcher_createBFMatcher(cv2.NORM_HAMMING)
            matches = bf.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)

            # Visualize matches on CPU
            img_matching = cv2.drawMatches(prev_img_cpu, kp1, img_cpu, kp2, matches[:100], None)
            img_matching = cv2.resize(img_matching, (img_matching.shape[1] // 2, img_matching.shape[0] // 2))
            cv2.imshow('feature matching CUDA', img_matching)
            cv2.waitKey(1)

            pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
            pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

            E, mask = cv2.findEssentialMat(pts1, pts2, focal=fx, pp=(cx, cy), method=cv2.RANSAC, prob=0.999,
                                           threshold=1.0)
            pts1 = pts1[mask.ravel() == 1]
            pts2 = pts2[mask.ravel() == 1]
            _, R, t, mask = cv2.recoverPose(E, pts1, pts2, focal=fx, pp=(cx, cy))

            R = R.T
            t = -np.dot(R, t)

            if i == 1:
                curr_R = R
                curr_t = t
                init_tx, init_tz = curr_t[0].item(), curr_t[2].item()
            else:
                curr_R = np.dot(prev_R, R)
                curr_t = np.dot(prev_R, t) + prev_t

            # Draw keypoints
            curr_img_kp = cv2.drawKeypoints(img_cpu, kp2, None, color=(0, 255, 0))
            cv2.imshow('keypoints from current image CUDA', curr_img_kp)

        tx, ty, tz = curr_t[0], curr_t[1], curr_t[2]
        qw, qx, qy, qz = rot2quat(curr_R)

        # Calculate processing time for the current frame
        timestamp = time.time() - start_time
        processing_times.append(timestamp)

        with open(out_pose_file, 'a') as f:
            f.write('%f %f %f %f %f %f %f %f\n' % (
                float(timestamp), float(tx.item()), float(ty.item()), float(tz.item()),
                float(qx.item()), float(qy.item()), float(qz.item()), float(qw.item())
            ))

        prev_R = curr_R
        prev_t = curr_t

        offset_draw = args.len_trajMap // 2
        offset_draw_small = args.len_trajMap // 8

        if args.dataset_type == 'TUM':
            cv2.circle(trajMap,
                       (int(tx + offset_draw), int(tz + offset_draw_small)),
                       1, (255, 0, 0), 2)
        elif args.dataset_type == 'KITTI':
            cv2.circle(trajMap,
                       (int(tx + offset_draw), int(tz + offset_draw_small)),
                       1, (255, 0, 0), 2)
        else:
            # Normalize and scale coordinates to draw
            draw_x = int((tx.item()) * 10 + offset_draw)
            draw_z = int((tz.item()) * 10 + offset_draw_small)

            # Boundary check
            if 0 <= draw_x < args.len_trajMap and 0 <= draw_z < args.len_trajMap:
                cv2.circle(trajMap, (draw_x, draw_z), 1, (255, 0, 0), 2)

        trajMap1 = cv2.flip(trajMap, 0)
        cv2.imshow('Trajectory CUDA', trajMap1)
        cv2.waitKey(1)

    # Calculate and display average FPS
    avg_processing_time = np.mean(processing_times)
    avg_fps = 1.0 / avg_processing_time if avg_processing_time > 0 else 0
    print(f"Average processing time per frame: {avg_processing_time:.4f} seconds")
    print(f"Average FPS: {avg_fps:.2f}")

    traj_img_file = f'{args.dataset_type}-{seq}_trajMap.png'
    cv2.imwrite(traj_img_file, trajMap1)
'''


Processing frames: 100%|██████████| 1591/1591 [06:23<00:00,  4.15frame/s] =>0

Processing frames:  58%|█████▊    | 920/1591 [03:16<02:22,  4.69frame/s] => 0
'''
