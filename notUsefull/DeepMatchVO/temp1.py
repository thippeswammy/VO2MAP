import matplotlib.pyplot as plt
import numpy as np


def load_poses(file_path):
    poses = []
    with open(file_path, 'r') as f:
        for line in f:
            values = list(map(float, line.strip().split()))
            pose = np.array(values).reshape(3, 4)  # 3x4 matrix
            poses.append(pose)
    return np.array(poses)


def plot_trajectory(poses):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x = poses[:, 0, 3]
    y = poses[:, 1, 3]
    z = poses[:, 2, 3]

    ax.plot(x, y, z, label='Predicted Trajectory', color='blue')
    ax.set_title('Full Predicted Trajectory - KITTI Seq 09')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    ax.grid(True)
    plt.show()


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--pose_file', type=str, required=True, help='Path to *_full.txt pose file')
    # args = parser.parse_args()

    poses = load_poses(
        r"F:\RunningProjects\VisualOdemetry\Visual-odometry-tutorial\notUsefull\DeepMatchVO\DeepMatchVO\output\full_pose\09_full.txt")
    plot_trajectory(poses)
