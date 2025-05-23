import argparse
import time
from glob import glob

import cv2
from tqdm import tqdm

from pose_evaluation_utils import *

print('cuda.getCudaEnabledDeviceCount()', cv2.cuda.getCudaEnabledDeviceCount())

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default='../Data/',
                    help="dataset root")
parser.add_argument("--dataset_type", type=str, default='CUSTOM', choices=['KITTI', 'CUSTOM'],
                    help="dataset type")
parser.add_argument("--len_trajMap", type=int, default=700,
                    help="size of the trajectory map")
args = parser.parse_args()

# dataset directory
if args.dataset_type == 'CUSTOM':
    seq = '02'
    img_data_dir = args.data_dir + 'CUSTOM/' + seq
elif args.dataset_type == 'KITTI':
    seq = '01'
    img_data_dir = args.data_dir + 'KITTI/' + seq
Visualize = False
# intrinsic parameters
if args.dataset_type == 'CUSTOM':
    width = 1920.0
    height = 880.0
    fx, fy, cx, cy = [1101.86382, 1114.96528, 885.646472, 443.776090]
elif args.dataset_type == 'KITTI':
    width = 1241.0
    height = 376.0
    fx, fy, cx, cy = [718.8560, 718.8560, 607.1928, 185.2157]

if __name__ == "__main__":
    # define the output pose file
    trajMap = trajMap1 = np.zeros((args.len_trajMap, args.len_trajMap, 3), dtype=np.uint8)
    if args.dataset_type == 'CUSTOM':
        out_pose_file = './Results/' + 'CUSTOM' + '-' + seq + '-traj_est.txt'
    elif args.dataset_type == 'KITTI':
        out_pose_file = './Results/' + 'KITTI' + '-' + seq + '-traj_est.txt'

    # get the image list in the directory
    img_list = glob(img_data_dir + '/*.png')
    img_list.sort()
    num_frames = len(img_list)

    # List to store processing times for each frame
    processing_times = []

    # Wrap the loop with tqdm for progress bar
    for i in tqdm(range(num_frames), desc="Processing frames", unit="frame"):
        start_time = time.time()
        # Load image
        img_cpu = cv2.imread(img_list[i], 0)
        cv2.imshow("Current Image", img_cpu)
        curr_img = cv2.cuda_GpuMat()
        curr_img.upload(img_cpu)
        if i == 0:
            curr_R = np.eye(3)
            curr_t = np.array([0, 0, 0])
        else:
            prev_img_cpu = cv2.imread(img_list[i - 1], 0)
            prev_img = cv2.cuda_GpuMat()
            prev_img.upload(prev_img_cpu)
            # ====================== Use ORB Feature to do feature matching =====================#
            #  Create ORB features
            orb = cv2.cuda_ORB.create(nfeatures=5000)
            kp1_gpu, des1 = orb.detectAndComputeAsync(prev_img, None)
            kp2_gpu, des2 = orb.detectAndComputeAsync(curr_img, None)

            kp1 = orb.convert(kp1_gpu)
            kp2 = orb.convert(kp2_gpu)

            # use brute-force matcher
            # BFMatcher (GPU)
            bf = cv2.cuda.DescriptorMatcher_createBFMatcher(cv2.NORM_HAMMING)
            matches = bf.match(des1, des2)

            # Sort the matched key points in the order of matching distance
            # so the best matches came to the front
            matches = sorted(matches, key=lambda x: x.distance)

            # Visualize matches on CPU
            if Visualize:
                img_matching = cv2.drawMatches(prev_img_cpu, kp1, img_cpu, kp2, matches[:100], None)
                img_matching = cv2.resize(img_matching, (img_matching.shape[1] // 2, img_matching.shape[0] // 2))
                cv2.imshow('feature matching CUDA', img_matching)
                cv2.waitKey(1)

            pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
            pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

            # compute essential matrix
            E, mask = cv2.findEssentialMat(pts1, pts2, focal=fx, pp=(cx, cy), method=cv2.RANSAC, prob=0.95,
                                           threshold=1.0)
            pts1 = pts1[mask.ravel() == 1]
            pts2 = pts2[mask.ravel() == 1]
            _, R, t, mask = cv2.recoverPose(E, pts1, pts2, focal=fx, pp=(cx, cy))

            # get camera motion
            R = R.T
            t = -np.dot(R, t)

            if i == 1:
                curr_R = R
                curr_t = t
                init_tx, init_tz = curr_t[0].item(), curr_t[2].item()
            else:
                curr_R = np.dot(prev_R, R)
                curr_t = np.dot(prev_R, t) + prev_t

            # draw the current image with keypoints
            if Visualize:
                curr_img_kp = cv2.drawKeypoints(img_cpu, kp2, None, color=(0, 255, 0), flags=0)
                cv2.imshow('key points from current image', curr_img_kp)

        # save current pose
        [tx, ty, tz] = [curr_t[0], curr_t[1], curr_t[2]]
        qw, qx, qy, qz = rot2quat(curr_R)

        end_time = time.time()
        timestamp = end_time - start_time
        processing_times.append(timestamp)

        with open(out_pose_file, 'a') as f:
            f.write('%f %f %f %f %f %f %f %f\n' % (
                float(timestamp), float(tx.item()), float(ty.item()), float(tz.item()),
                float(qx.item()), float(qy.item()), float(qz.item()), float(qw.item())
            ))

        prev_R = curr_R
        prev_t = curr_t

        # draw estimated trajectory (blue)
        offset_draw = (int(args.len_trajMap / 2))
        offset_draw_small = (int(args.len_trajMap / 8))

        # Different starting points depending on the dataset
        if args.dataset_type == 'CUSTOM':
            cv2.circle(trajMap,
                       (int(float(curr_t[0]) + offset_draw), int(float(curr_t[2]) + offset_draw_small)),
                       1, (255, 0, 0), 2)
        elif args.dataset_type == 'KITTI':
            cv2.circle(trajMap, (int(curr_t[0] + offset_draw), int(curr_t[2]) + offset_draw_small), 1, (255, 0, 0), 2)

        trajMap1 = cv2.flip(trajMap, 0)
        trajMap1 = cv2.resize(trajMap1, (700, 700))
        cv2.imshow('Trajectory', trajMap1)
        cv2.waitKey(1)

    # Calculate and display average FPS
    avg_processing_time = np.mean(processing_times)
    avg_fps = 1.0 / avg_processing_time if avg_processing_time > 0 else 0
    print(f"Average processing time per frame: {avg_processing_time:.4f} seconds")
    print(f"Average FPS: {avg_fps:.2f}")

    # Save the trajectory result depending on the dataset
    if args.dataset_type == 'CUSTOM':
        cv2.imwrite(('Results/CUSTOM' + '-' + seq + '_trajMap.png'), trajMap1)
    elif args.dataset_type == 'KITTI':
        cv2.imwrite('Results/KITTI' + '-' + seq + '_trajMap.png', trajMap1)
