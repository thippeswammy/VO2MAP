import numpy as np
from scipy.spatial.transform import Rotation as R


def convert_colmap_images_to_kitti(images_txt_path, out_pose_file, scale=1.0):
    with open(images_txt_path, 'r') as f:
        lines = f.readlines()

    with open(out_pose_file, 'w') as out_f:
        for line in lines:
            if line.strip() == "" or line.startswith("#"):
                continue
            parts = line.strip().split()
            if len(parts) < 8:
                continue  # not a valid line

            # Parse quaternion and translation
            qw, qx, qy, qz = map(float, parts[1:5])
            tx, ty, tz = map(float, parts[5:8])
            tx, ty, tz = np.array([tx, ty, tz]) * scale

            # Convert quaternion to rotation matrix
            R_mat = R.from_quat([qx, qy, qz, qw]).as_matrix()

            # Compose 3x4 transformation matrix
            T = np.hstack((R_mat, np.array([[tx], [ty], [tz]])))

            # Flatten row-major and write
            pose_line = ' '.join(map(str, T.flatten()))
            out_f.write(pose_line + "\n")

    print(f"Saved KITTI format poses to: {out_pose_file}")


# Example usage:
input_file = (
    r'F:\RunningProjects\VisualOdemetry\Visual-odometry-tutorial\working\kitti-odom-eval\result\example_5\09_1.txt')  # or Path("your_path/poses.txt")
output_file = (
    r'F:\RunningProjects\VisualOdemetry\Visual-odometry-tutorial\working\kitti-odom-eval\result\example_5\09.txt')
convert_colmap_images_to_kitti(input_file, output_file)
