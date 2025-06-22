# Copyright (C) Huangying Zhan 2019. All rights reserved.

import numpy as np


class KittiEvalOdom:
    """Evaluate odometry result for t_rel, r_rel, and t_abs (ATE)"""

    def __init__(self):
        # Segment lengths for relative error evaluation (in meters)
        self.lengths = [100]
        self.step_size = 1  # Step size for selecting first frame of segments

    def load_poses_from_txt(self, file_name):
        """Load poses from txt (KITTI format)
        Args:
            file_name (str): Path to txt file
        Returns:
            poses (dict): {idx: 4x4 array}
        """
        poses = {}
        with open(file_name, 'r') as f:
            for cnt, line in enumerate(f):
                line_split = [float(i) for i in line.split() if i != ""]
                P = np.eye(4)
                for row in range(3):
                    for col in range(4):
                        P[row, col] = line_split[row * 4 + col]
                poses[cnt] = P
        return poses

    def trajectory_distances(self, poses):
        """Compute distance for each pose w.r.t frame-0
        Args:
            poses (dict): {idx: 4x4 array}
        Returns:
            dist (list): Distance of each pose w.r.t frame-0
        """
        dist = [0]
        sort_frame_idx = sorted(poses.keys())
        for i in range(len(sort_frame_idx) - 1):
            cur_frame = sort_frame_idx[i]
            next_frame = sort_frame_idx[i + 1]
            P1 = poses[cur_frame]
            P2 = poses[next_frame]
            dx = P1[0, 3] - P2[0, 3]
            dy = P1[1, 3] - P2[1, 3]
            dz = P1[2, 3] - P2[2, 3]
            dist.append(dist[i] + np.sqrt(dx ** 2 + dy ** 2 + dz ** 2))
        return dist

    def rotation_error(self, pose_error):
        """Compute rotation error
        Args:
            pose_error (4x4 array): Relative pose error
        Returns:
            rot_error (float): Rotation error in radians
        """
        a = pose_error[0, 0]
        b = pose_error[1, 1]
        c = pose_error[2, 2]
        d = 0.5 * (a + b + c - 1.0)
        return np.arccos(max(min(d, 1.0), -1.0))

    def translation_error(self, pose_error):
        """Compute translation error
        Args:
            pose_error (4x4 array): Relative pose error
        Returns:
            trans_error (float): Translation error in meters
        """
        dx = pose_error[0, 3]
        dy = pose_error[1, 3]
        dz = pose_error[2, 3]
        return np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

    def last_frame_from_segment_length(self, dist, first_frame, length):
        """Find frame index that is away from first_frame by required distance
        Args:
            dist (list): Distances w.r.t frame-0
            first_frame (int): Start frame index
            length (float): Required segment length
        Returns:
            int: End frame index or -1 if not found
        """
        for i in range(first_frame, len(dist)):
            if dist[i] > (dist[first_frame] + length):
                return i
        return -1

    def calc_sequence_errors(self, poses_gt, poses_result):
        """Calculate sequence errors for t_rel and r_rel
        Args:
            poses_gt (dict): {idx: 4x4 array}, ground truth poses
            poses_result (dict): {idx: 4x4 array}, predicted poses
        Returns:
            err (list): [first_frame, rot_err/len, trans_err/len, len]
        """
        err = []
        dist = self.trajectory_distances(poses_gt)
        for first_frame in range(0, len(poses_gt), self.step_size):
            for len_ in self.lengths:
                last_frame = self.last_frame_from_segment_length(dist, first_frame, len_)
                if last_frame == -1 or last_frame not in poses_result or first_frame not in poses_result:
                    continue
                # Compute relative pose error
                pose_delta_gt = np.dot(np.linalg.inv(poses_gt[first_frame]), poses_gt[last_frame])
                pose_delta_result = np.dot(np.linalg.inv(poses_result[first_frame]), poses_result[last_frame])
                pose_error = np.dot(np.linalg.inv(pose_delta_result), pose_delta_gt)
                r_err = self.rotation_error(pose_error)
                t_err = self.translation_error(pose_error)
                err.append([first_frame, r_err / len_, t_err / len_, len_])
        return err

    def compute_overall_err(self, seq_err):
        """Compute average t_rel and r_rel
        Args:
            seq_err (list): [first_frame, rot_err/len, trans_err/len, len]
        Returns:
            ave_t_err (float): Average translation error (%)
            ave_r_err (float): Average rotation error (rad/m)
        """
        t_err = r_err = 0
        seq_len = len(seq_err)
        if seq_len > 0:
            for item in seq_err:
                r_err += item[1]
                t_err += item[2]
            return t_err / seq_len, r_err / seq_len
        return 0, 0

    def compute_ATE(self, gt, pred):
        """Compute Absolute Translation Error (t_abs)
        Args:
            gt (dict): {idx: 4x4 array}, ground truth poses
            pred (dict): {idx: 4x4 array}, predicted poses
        Returns:
            ate (float): RMSE of absolute translation errors (meters)
        """
        errors = []
        for i in pred:
            gt_xyz = gt[i][:3, 3]
            pred_xyz = pred[i][:3, 3]
            align_err = gt_xyz - pred_xyz
            errors.append(np.sqrt(np.sum(align_err ** 2)))
        return np.sqrt(np.mean(np.asarray(errors) ** 2))

    def scale_lse_solver(self, X, Y):
        """Compute optimal scaling factor
        Args:
            X (KxN array): Predicted data
            Y (KxN array): Ground truth data
        Returns:
            scale (float): Scaling factor
        """
        return np.sum(X * Y) / np.sum(X ** 2)

    def scale_optimization(self, gt, pred):
        """Optimize scaling factor for predicted poses
        Args:
            gt (dict): Ground truth poses
            pred (dict): Predicted poses
        Returns:
            pred_updated (dict): Scaled predicted poses
        """
        pred_updated = pred.copy()
        xyz_pred = np.array([pred[i][:3, 3] for i in pred])
        xyz_ref = np.array([gt[i][:3, 3] for i in gt])
        scale = self.scale_lse_solver(xyz_pred, xyz_ref)
        for i in pred_updated:
            pred_updated[i][:3, 3] *= scale
        return pred_updated

    def umeyama_alignment(self, x, y, with_scale=True):
        """Umeyama alignment for 6DOF or 7DOF
        Args:
            x (mxn array): Predicted points
            y (mxn array): Ground truth points
            with_scale (bool): Include scale if True
        Returns:
            r (array): Rotation matrix
            t (array): Translation vector
            c (float): Scale factor
        """
        if x.shape != y.shape:
            raise ValueError("x.shape must equal y.shape")
        m, n = x.shape
        mean_x = x.mean(axis=1)
        mean_y = y.mean(axis=1)
        sigma_x = 1.0 / n * (np.linalg.norm(x - mean_x[:, np.newaxis]) ** 2)
        outer_sum = np.zeros((m, m))
        for i in range(n):
            outer_sum += np.outer((y[:, i] - mean_y), (x[:, i] - mean_x))
        cov_xy = outer_sum / n
        u, d, v = np.linalg.svd(cov_xy)
        s = np.eye(m)
        if np.linalg.det(u) * np.linalg.det(v) < 0.0:
            s[m - 1, m - 1] = -1
        r = u.dot(s).dot(v)
        c = 1 / sigma_x * np.trace(np.diag(d).dot(s)) if with_scale else 1.0
        t = mean_y - c * r.dot(mean_x)
        return r, t, c

    def eval(self, gt_file, result_file, alignment=None):
        """Evaluate a single sequence
        Args:
            gt_file (str): Path to ground truth pose file
            result_file (str): Path to predicted pose file
            alignment (str): None, 'scale', '6dof', '7dof', or 'scale_7dof'
        Returns:
            dict: {t_rel (%), r_rel (deg/100m), t_abs (m)}
        """
        # Load poses
        poses_gt = self.load_poses_from_txt(gt_file)
        poses_result = self.load_poses_from_txt(result_file)

        # Align to first frame
        idx_0 = sorted(poses_result.keys())[0]
        pred_0 = poses_result[idx_0]
        gt_0 = poses_gt[idx_0]
        for cnt in poses_result:
            poses_result[cnt] = np.linalg.inv(pred_0) @ poses_result[cnt]
            poses_gt[cnt] = np.linalg.inv(gt_0) @ poses_gt[cnt]

        # Apply alignment if specified
        if alignment == "scale":
            poses_result = self.scale_optimization(poses_gt, poses_result)
        elif alignment in ["6dof", "7dof", "scale_7dof"]:
            xyz_gt = np.array([poses_gt[cnt][:3, 3] for cnt in poses_result]).T
            xyz_result = np.array([poses_result[cnt][:3, 3] for cnt in poses_result]).T
            r, t, scale = self.umeyama_alignment(xyz_result, xyz_gt, alignment != "6dof")
            align_transformation = np.eye(4)
            align_transformation[:3, :3] = r
            align_transformation[:3, 3] = t
            for cnt in poses_result:
                poses_result[cnt][:3, 3] *= scale
                if alignment in ["6dof", "7dof"]:
                    poses_result[cnt] = align_transformation @ poses_result[cnt]

        # Compute sequence errors for t_rel and r_rel
        seq_err = self.calc_sequence_errors(poses_gt, poses_result)
        ave_t_err, ave_r_err = self.compute_overall_err(seq_err)

        # Compute t_abs (ATE)
        ate = self.compute_ATE(poses_gt, poses_result)

        # Convert to required units
        t_rel = ave_t_err * 100  # Percentage
        r_rel = ave_r_err / np.pi * 180 * 100  # Degrees per 100m
        t_abs = ate  # Meters

        return {"t_rel (%)": t_rel, "r_rel (deg/100m)": r_rel, "t_abs (m)": t_abs}


# Example usage
if __name__ == "__main__":
    evaluator = KittiEvalOdom()
    gt_file = r"F:\RunningProjects\VisualOdemetry\Visual-odometry-tutorial\working\kitti-odom-eval\dataset\kitti_odom\gt_poses\09.txt"
    result_file = r"F:\RunningProjects\VisualOdemetry\Visual-odometry-tutorial\working\kitti-odom-eval\result\example_1\09.txt"
    alignment = "None"  # Options: None, "scale", "6dof", "7dof", "scale_7dof"
    results = evaluator.eval(gt_file, result_file, alignment)
    print(f"t_rel (%): {results['t_rel (%)']:.3f}")
    print(f"r_rel (deg/100m): {results['r_rel (deg/100m)']:.3f}")
    print(f"t_abs (m): {results['t_abs (m)']:.3f}")
