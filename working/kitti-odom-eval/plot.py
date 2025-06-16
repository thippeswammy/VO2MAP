import numpy as np
import plotly.graph_objects as go


def load_kitti_poses(file_path):
    """
    Load trajectory from a KITTI format file (each line: 3x4 transformation matrix row-major).
    Returns a list of translation vectors.
    """
    poses = []
    with open(file_path, 'r') as f:
        for line in f:
            T = np.fromstring(line.strip(), sep=' ').reshape(3, 4)
            poses.append(T[:, 3])  # Extract translation (tx, ty, tz)
    return np.array(poses)


def plot_kitti_trajectory_3d(pred_file, gt_file=None, title="3D Trajectory", save_path=None):
    """
    Plots predicted (and optionally ground truth) trajectory in 3D using Plotly for interactivity.
    """
    # Load predicted trajectory
    pred_xyz = load_kitti_poses(pred_file)

    # Create Plotly scatter plot for predicted trajectory
    pred_trace = go.Scatter3d(
        x=pred_xyz[:, 0], y=pred_xyz[:, 1], z=pred_xyz[:, 2],
        mode='lines+markers',
        name='Predicted',
        line=dict(color='blue', width=2),
        marker=dict(size=3)
    )

    # Initialize list of traces
    data = [pred_trace]

    # Load and plot ground truth if provided
    if gt_file:
        gt_xyz = load_kitti_poses(gt_file)
        gt_trace = go.Scatter3d(
            x=gt_xyz[:, 0], y=gt_xyz[:, 1], z=gt_xyz[:, 2],
            mode='lines+markers',
            name='Ground Truth',
            line=dict(color='gray', width=2, dash='dash'),
            marker=dict(size=3)
        )
        data.append(gt_trace)

    # Set layout for the plot
    layout = go.Layout(
        title=title,
        scene=dict(
            xaxis_title='X [m]',
            yaxis_title='Y [m]',
            zaxis_title='Z [m]',
            aspectmode='cube',  # Ensure equal aspect ratio
            xaxis=dict(gridcolor='rgb(200, 200, 200)', zerolinecolor='black'),
            yaxis=dict(gridcolor='rgb(200, 200, 200)', zerolinecolor='black'),
            zaxis=dict(gridcolor='rgb(200, 200, 200)', zerolinecolor='black')
        ),
        showlegend=True,
        margin=dict(l=0, r=0, b=0, t=40)
    )

    # Compute axis limits to ensure equal scaling
    all_pts = np.vstack([pred_xyz] + ([gt_xyz] if gt_file else []))
    min_vals, max_vals = all_pts.min(axis=0), all_pts.max(axis=0)
    max_range = (max_vals - min_vals).max() / 2
    mid_vals = (max_vals + min_vals) / 2
    layout.scene.update(
        xaxis_range=[mid_vals[0] - max_range, mid_vals[0] + max_range],
        yaxis_range=[mid_vals[1] - max_range, mid_vals[1] + max_range],
        zaxis_range=[mid_vals[2] - max_range, mid_vals[2] + max_range]
    )

    # Create figure
    fig = go.Figure(data=data, layout=layout)

    # Save or show the plot
    if save_path:
        fig.write_html(save_path)
        print(f"Saved interactive 3D plot to {save_path}")
    else:
        fig.show()


# Example usage
plot_kitti_trajectory_3d("result/example_5/09.txt", gt_file="results/09.txt", title="3D Trajectory: Predicted vs GT",
                         save_path="traj_3d_plot.html")
