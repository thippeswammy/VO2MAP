import matplotlib
import numpy as np

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Load poses
pose_file_path = './custom-custom-traj_est.txt'
poses = np.loadtxt(pose_file_path)

# Ensure poses has at least 4 columns (for x and z)
if poses.ndim == 1:
    poses = poses.reshape(1, -1)

if poses.shape[1] < 4:
    raise ValueError(f"Expected at least 4 columns in pose file, got {poses.shape[1]}")

# Extract x and z translation, safely
try:
    x = poses[:, 1].astype(np.float64)
    z = poses[:, 3].astype(np.float64)
except Exception as e:
    print("Error converting x or z to float:", e)
    print("Sample x:", poses[:, 1][:5])
    print("Sample z:", poses[:, 3][:5])
    raise

# Filter out any NaNs or invalid values
valid_mask = np.isfinite(x) & np.isfinite(z)
x = x[valid_mask]
z = z[valid_mask]

# Create plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x, z, marker='o', linestyle='-', color='blue', label='Trajectory')
ax.set_xlabel('X Position')
ax.set_ylabel('Z Position')
ax.set_title('Camera Trajectory (X-Z Plane)')
ax.legend()
ax.grid(True)
ax.axis('equal')

# Show plot
plt.tight_layout()
plt.show()
