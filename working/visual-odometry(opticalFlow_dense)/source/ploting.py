import matplotlib.pyplot as plt
import numpy as np


def plot_2d_trajectories(estimated, groundtruth, index):
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"2D Projections of Trajectory Comparison [Index {index}]", fontsize=14)

    # XY Plot
    axs[0].plot(estimated[:, 0], estimated[:, 1], 'b-', label='Estimated')
    axs[0].plot(groundtruth[:, 0], groundtruth[:, 1], 'g--', label='Groundtruth')
    axs[0].set_xlabel("X (m)")
    axs[0].set_ylabel("Y (m)")
    axs[0].set_title("XY Plane")
    axs[0].legend()
    axs[0].grid(True)

    # XZ Plot
    axs[1].plot(estimated[:, 0], estimated[:, 2], 'b-', label='Estimated')
    axs[1].plot(groundtruth[:, 0], groundtruth[:, 2], 'g--', label='Groundtruth')
    axs[1].set_xlabel("X (m)")
    axs[1].set_ylabel("Z (m)")
    axs[1].set_title("XZ Plane")
    axs[1].legend()
    axs[1].grid(True)

    # YZ Plot
    axs[2].plot(estimated[:, 1], estimated[:, 2], 'b-', label='Estimated')
    axs[2].plot(groundtruth[:, 1], groundtruth[:, 2], 'g--', label='Groundtruth')
    axs[2].set_xlabel("Y (m)")
    axs[2].set_ylabel("Z (m)")
    axs[2].set_title("YZ Plane")
    axs[2].legend()
    axs[2].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"trajectory_2d_projections_{index}.png", dpi=300)
    plt.show()


# Loop through 6 datasets
for i in range(0, 1):
    estimated_aligned = np.load('estimated_aligned.npy')
    kitti_positions = np.load('kitti_positions.npy')

    # You may want to select a slice for each i if estimated_aligned contains multiple sets
    # For example: estimated_aligned[i] if shape is (6, N, 3)
    # Modify the line below if needed based on actual shape
    plot_2d_trajectories(estimated_aligned, kitti_positions, index=i)
