import matplotlib.pyplot as plt
import numpy as np


def plot_multiple_trajectories(gt_file, result_files, labels, save_path):
    """
    Plot multiple predicted trajectories along with ground truth.

    Args:
        gt_file (str): path to ground truth .txt file
        result_files (list of str): list of paths to predicted pose .txt files
        labels (list of str): list of labels for each result (same length as result_files)
        save_path (str): where to save the output .png plot
    """

    def load_poses_from_txt(file_path):
        poses = {}
        with open(file_path, 'r') as f:
            for i, line in enumerate(f):
                values = list(map(float, line.strip().split()))
                pose = np.array(values).reshape(3, 4)
                pose = np.vstack((pose, [0, 0, 0, 1]))
                poses[i] = pose
        return poses

    fontsize_ = 16
    plt.figure()
    ax = plt.gca()
    ax.set_aspect('equal')

    # Load and plot GT
    poses_gt = load_poses_from_txt(gt_file)
    pos_gt = [[pose[0, 3], pose[2, 3]] for pose in poses_gt.values()]
    pos_gt = np.array(pos_gt)
    plt.plot(pos_gt[:, 0], pos_gt[:, 1], label="Ground Truth", linewidth=2)

    # Load and plot each prediction
    for result_file, label in zip(result_files, labels):
        poses_pred = load_poses_from_txt(result_file)
        pos_pred = [[pose[0, 3], pose[2, 3]] for pose in poses_pred.values()]
        pos_pred = np.array(pos_pred)
        plt.plot(pos_pred[:, 0], pos_pred[:, 1], label=label)

    plt.legend(loc="upper right", prop={'size': fontsize_})
    plt.xticks(fontsize=fontsize_)
    plt.yticks(fontsize=fontsize_)
    plt.xlabel('x (m)', fontsize=fontsize_)
    plt.ylabel('z (m)', fontsize=fontsize_)
    plt.title("Trajectory Comparison", fontsize=fontsize_ + 2)
    plt.grid(True)
    plt.gcf().set_size_inches(10, 10)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()


gt = "F:\RunningProjects\VisualOdemetry\Visual-odometry-tutorial\data\kitti-odom\09\poses.txt"
results = [
    r"F:\RunningProjects\VisualOdemetry\Visual-odometry-tutorial\working\visual-odometry(opticalFlow_dense)\source\KITTI-09-traj_est.txt",
    r"F:\RunningProjects\VisualOdemetry\Visual-odometry-tutorial\working\visual-odometry(opticalFlow_dense)\source\KITTI-09-traj_est_by_vo.txt",
    # r"path/to/results/09_methode3.txt"
]
labels = ["Method 1", "Method 2"]
output = "trajectory_comparison_seq09.png"

plot_multiple_trajectories(gt, results, labels, output)
