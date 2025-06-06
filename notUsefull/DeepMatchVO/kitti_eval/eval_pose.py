from __future__ import division

import argparse
import os
from glob import glob

import matplotlib.pyplot as plt
import numpy as np

from pose_evaluation_utils import compute_ate

parser = argparse.ArgumentParser()
parser.add_argument("--gtruth_dir", type=str, help="Path to the directory with ground-truth trajectories")
parser.add_argument("--pred_dir", type=str, help="Path to the directory with predicted trajectories")
args = parser.parse_args()


def read_pose_file(filepath):
    traj = []
    with open(filepath, 'r') as f:
        for line in f:
            vals = list(map(float, line.strip().split()))
            if len(vals) == 12:
                T = np.eye(4)
                T[:3, :] = np.array(vals).reshape(3, 4)
                traj.append(T[:3, 3])
            elif len(vals) == 8:
                traj.append([vals[1], vals[2], vals[3]])
    return np.array(traj)


def main():
    # pred_files = sorted(glob(args.pred_dir + '/*.txt'))
    pred_files = sorted(
        glob(r'F:\RunningProjects\VisualOdemetry\Visual-odometry-tutorial\notUsefull\DeepMatchVO\output\09' + '/*.txt'))
    all_pred = []
    all_gt = []
    ate_all = []

    for pred_file in pred_files:
        fname = os.path.basename(pred_file)
        # gt_file = os.path.join(args.gtruth_dir, fname)
        gt_file = os.path.join(
            r'F:\RunningProjects\VisualOdemetry\Visual-odometry-tutorial\notUsefull\DeepMatchVO\kitti_eval\pose_data\ground_truth\seq3\09',
            fname)
        print('gt_file=>')
        if not os.path.exists(gt_file):
            print("Missing ground-truth for:", fname)
            continue

        ate = compute_ate(gt_file, pred_file)
        if ate is False:
            continue
        ate_all.append(ate)

        pred_traj = read_pose_file(pred_file)
        gt_traj = read_pose_file(gt_file)

        all_pred.append(pred_traj)
        all_gt.append(gt_traj)

    if not all_pred or not all_gt:
        print("No valid data found.")
        return

    # Concatenate all segments into full trajectory
    all_pred = np.vstack(all_pred)
    all_gt = np.vstack(all_gt)

    # Plot full trajectory
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(all_gt[:, 0], all_gt[:, 1], all_gt[:, 2], label='Ground Truth', c='green')
    ax.plot(all_pred[:, 0], all_pred[:, 1], all_pred[:, 2], label='Predicted', c='red')
    ax.set_title('Full Trajectory: Ground Truth vs Predicted')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.tight_layout()
    plt.show()

    # Print overall ATE
    ate_all = np.array(ate_all)
    print("ATE mean: %.4f, std: %.4f" % (np.mean(ate_all), np.std(ate_all)))


if __name__ == '__main__':
    main()

'''
python .\kitti_eval\eval_pose.py --gtruth_dir kitti_data/odometry/09.txt --pred_dir output\\full_pose\\09_full.txt
'''
