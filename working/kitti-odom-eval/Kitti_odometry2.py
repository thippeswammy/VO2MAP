# Copyright (C) Huangying Zhan 2019. All rights reserved.

import argparse
import copy
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from tabulate import tabulate


def scale_lse_solver(X, Y):
    """Least-square-error solver for scaling factor."""
    scale = np.sum(X * Y) / np.sum(X ** 2)
    return scale


def umeyama_alignment(x, y, with_scale=False):
    """Computes Sim(m) transformation parameters (Umeyama, 1991)."""
    if x.shape != y.shape:
        raise ValueError("x.shape must equal y.shape")
    m, n = x.shape
    mean_x = x.mean(axis=1)
    mean_y = y.mean(axis=1)
    sigma_x = 1.0 / n * (np.linalg.norm(x - mean_x[:, np.newaxis]) ** 2)
    outer_sum = np.zeros((m, m))
    for i in range(n):
        outer_sum += np.outer(y[:, i] - mean_y, x[:, i] - mean_x)
    cov_xy = outer_sum / n
    u, d, v = np.linalg.svd(cov_xy)
    s = np.eye(m)
    if np.linalg.det(u) * np.linalg.det(v) < 0.0:
        s[m - 1, m - 1] = -1
    r = u.dot(s).dot(v)
    c = 1 / sigma_x * np.trace(np.diag(d).dot(s)) if with_scale else 1.0
    t = mean_y - c * r.dot(mean_x)
    return r, t, c


def parse_computational_metrics(file_path):
    """Parse computational metrics from txt file."""
    metrics = {'total_time': 0.0, 'frames_processed': 0, 'fps': 0.0,
               'avg_cpu': 0.0, 'avg_cpu_overall': 0.0, 'avg_gpu': 0.0,
               'avg_gpu_mem': 0.0, 'avg_gpu_power': 0.0, 'avg_ram': 0.0,
               'monitor_duration': 0.0}
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if 'Total Time' in line:
                    metrics['total_time'] = float(line.split('=')[1].split()[0])
                elif 'Frames processed' in line:
                    metrics['frames_processed'] = int(line.split('=')[1].split()[0])
                elif 'FPS' in line:
                    metrics['fps'] = float(line.split('=')[1].split()[0])
                elif '[AVERAGE USAGE]' in line:
                    for subline in lines[lines.index(line) + 1:]:
                        if '}' in subline:
                            break
                        key, value = subline.split(':')
                        key = key.strip().replace("'", "").replace(" ", "_")
                        if key in metrics:
                            metrics[key] = float(value.split(',')[0].strip())
    except Exception as e:
        print(f"Error parsing computational metrics from {file_path}: {e}")
    return metrics


class KittiEvalOdom:
    """Evaluate odometry results for KITTI dataset."""

    def __init__(self):
        self.lengths = [100, 200, 300, 400, 500, 600, 700, 800]
        self.step_size = 10

    @staticmethod
    def plot_3d_trajectories_interactive(estimated, groundtruth, title="3D Trajectory Comparison",
                                         file_name="trajectory_3d_interactive.html"):
        """Plot interactive 3D trajectories using Plotly."""
        estimated = np.asarray(estimated, dtype=np.float32)
        groundtruth = np.asarray(groundtruth, dtype=np.float32)

        fig = go.Figure()
        fig.add_trace(go.Scatter3d(
            x=estimated[:, 0], y=estimated[:, 1], z=estimated[:, 2],
            mode='lines+markers',
            marker=dict(size=3, color='blue'),
            line=dict(color='blue'),
            name='Estimated'
        ))
        fig.add_trace(go.Scatter3d(
            x=groundtruth[:, 0], y=groundtruth[:, 1], z=groundtruth[:, 2],
            mode='lines+markers',
            marker=dict(size=3, color='green'),
            line=dict(color='green', dash='dash'),
            name='Groundtruth'
        ))
        fig.update_layout(
            title=title,
            scene=dict(xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='Z (m)'),
            legend=dict(x=0.02, y=0.98)
        )
        fig.write_html(file_name)

    @staticmethod
    def load_poses_from_txt(file_name):
        """Load poses from KITTI-format txt file."""
        try:
            with open(file_name, 'r') as f:
                lines = f.readlines()
            poses = {}
            for cnt, line in enumerate(lines):
                try:
                    line_split = [float(i) for i in line.strip().split() if i]
                    if len(line_split) not in [12, 13]:
                        raise ValueError(f"Invalid number of values in line {cnt + 1}")
                    P = np.eye(4)
                    offset = 1 if len(line_split) == 13 else 0
                    for row in range(3):
                        for col in range(4):
                            P[row, col] = line_split[row * 4 + col + offset]
                    frame_idx = int(line_split[0]) if offset else cnt
                    poses[frame_idx] = P
                except ValueError as e:
                    print(f"Error parsing line {cnt + 1} in {file_name}: {e}")
                    continue
            if not poses:
                raise ValueError(f"No valid poses found in {file_name}")
            return poses
        except Exception as e:
            raise Exception(f"Failed to load poses from {file_name}: {e}")

    @staticmethod
    def trajectory_distances(poses):
        """Compute distances w.r.t. frame 0."""
        dist = [0]
        keys = sorted(poses.keys())
        for i in range(len(keys) - 1):
            P1, P2 = poses[keys[i]], poses[keys[i + 1]]
            dx, dy, dz = P1[0, 3] - P2[0, 3], P1[1, 3] - P2[1, 3], P1[2, 3] - P2[2, 3]
            dist.append(dist[i] + np.sqrt(dx ** 2 + dy ** 2 + dz ** 2))
        return dist

    @staticmethod
    def rotation_error(pose_error):
        """Compute rotation error in radians."""
        a, b, c = pose_error[0, 0], pose_error[1, 1], pose_error[2, 2]
        d = 0.5 * (a + b + c - 1.0)
        return np.arccos(max(min(d, 1.0), -1.0))

    @staticmethod
    def translation_error(pose_error):
        """Compute translation error in meters."""
        dx, dy, dz = pose_error[0, 3], pose_error[1, 3], pose_error[2, 3]
        return np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

    @staticmethod
    def last_frame_from_segment_length(dist, first_frame, length):
        """Find frame index for segment of specified length."""
        for i in range(first_frame, len(dist)):
            if dist[i] > dist[first_frame] + length:
                return i
        return -1

    def calc_sequence_errors(self, poses_gt, poses_result):
        """Calculate relative translation and rotation errors."""
        err = []
        dist = self.trajectory_distances(poses_gt)
        for first_frame in range(0, len(poses_gt), self.step_size):
            for length in self.lengths:
                last_frame = self.last_frame_from_segment_length(dist, first_frame, length)
                if last_frame == -1 or last_frame not in poses_result or first_frame not in poses_result:
                    continue
                pose_delta_gt = np.linalg.inv(poses_gt[first_frame]) @ poses_gt[last_frame]
                pose_delta_result = np.linalg.inv(poses_result[first_frame]) @ poses_result[last_frame]
                pose_error = np.linalg.inv(pose_delta_result) @ pose_delta_gt
                r_err = self.rotation_error(pose_error)
                t_err = self.translation_error(pose_error)
                err.append([first_frame, r_err / length, t_err / length, length])
        return err

    @staticmethod
    def compute_overall_err(seq_err):
        """Compute average t_rel and r_rel."""
        if not seq_err:
            return 0, 0
        t_err = sum(item[2] for item in seq_err) / len(seq_err)
        r_err = sum(item[1] for item in seq_err) / len(seq_err)
        return t_err, r_err

    def compute_segment_error(self, seq_errs):
        """Calculate average errors for different segments."""
        segment_errs = {length: [] for length in self.lengths}
        avg_segment_errs = {}
        for err in seq_errs:
            length = err[3]
            t_err, r_err = err[2], err[1]
            segment_errs[length].append([t_err, r_err])
        for length in self.lengths:
            if segment_errs[length]:
                avg_t_err = np.mean(np.asarray(segment_errs[length])[:, 0])
                avg_r_err = np.mean(np.asarray(segment_errs[length])[:, 1])
                avg_segment_errs[length] = [avg_t_err, avg_r_err]
            else:
                avg_segment_errs[length] = []
        return avg_segment_errs

    @staticmethod
    def compute_ATE(gt, pred):
        """Compute Absolute Trajectory Error (RMSE)."""
        errors = []
        for i in pred:
            gt_xyz = gt[i][:3, 3]
            pred_xyz = pred[i][:3, 3]
            errors.append(np.sqrt(np.sum((gt_xyz - pred_xyz) ** 2)))
        return np.sqrt(np.mean(np.asarray(errors) ** 2)) if errors else 0

    def compute_RPE(self, gt, pred):
        """Compute Relative Pose Error (translation and rotation)."""
        trans_errors, rot_errors = [], []
        keys = sorted(pred.keys())[:-1]
        for i in keys:
            if i + 1 in pred and i + 1 in gt:
                gt_rel = np.linalg.inv(gt[i]) @ gt[i + 1]
                pred_rel = np.linalg.inv(pred[i]) @ pred[i + 1]
                rel_err = np.linalg.inv(gt_rel) @ pred_rel
                trans_errors.append(self.translation_error(rel_err))
                rot_errors.append(self.rotation_error(rel_err))
        return np.mean(trans_errors) if trans_errors else 0, np.mean(rot_errors) if rot_errors else 0

    def compute_translational_rmse(self, gt, pred):
        """Compute Translational RMSE (same as ATE for consistency)."""
        return self.compute_ATE(gt, pred)

    @staticmethod
    def scale_optimization(gt, pred):
        """Optimize scaling factor for predicted poses."""
        pred_updated = copy.deepcopy(pred)
        xyz_pred = np.array([pred[i][:3, 3] for i in pred])
        xyz_ref = np.array([gt[i][:3, 3] for i in gt])
        scale = scale_lse_solver(xyz_pred, xyz_ref)
        for i in pred_updated:
            pred_updated[i][:3, 3] *= scale
        return pred_updated

    @staticmethod
    def compute_total_distance(poses):
        """Compute total trajectory distance."""
        keys = sorted(poses.keys())
        dist = 0.0
        for i in range(1, len(keys)):
            dist += np.linalg.norm(poses[keys[i]][:3, 3] - poses[keys[i - 1]][:3, 3])
        return dist

    @staticmethod
    def compute_drift(gt, pred):
        """Compute final drift (Euclidean distance at last frame)."""
        keys = sorted(gt.keys())
        if keys and keys[-1] in pred:
            return np.linalg.norm(gt[keys[-1]][:3, 3] - pred[keys[-1]][:3, 3])
        return 0.0

    @staticmethod
    def write_result(f, seq, errs):
        """Write evaluation metrics to file."""
        t_rel, r_rel, ate, rpe_trans, rpe_rot, trans_rmse, gt_dist, pred_dist, drift = errs
        lines = [
            f"Sequence: \t {seq}\n",
            f"t_rel (%): \t {t_rel * 100:.3f}\n",
            f"r_rel (deg/100m): \t {r_rel / np.pi * 180 * 100:.3f}\n",
            f"ATE (m): \t {ate:.3f}\n",
            f"Translational RMSE (m): \t {trans_rmse:.3f}\n",
            f"RPE trans (m): \t {rpe_trans:.3f}\n",
            f"RPE rot (deg): \t {rpe_rot * 180 / np.pi:.3f}\n",
            f"GT Distance (m): \t {gt_dist:.3f}\n",
            f"Pred Distance (m): \t {pred_dist:.3f}\n",
            f"Drift (m): \t {drift:.3f}\n\n"
        ]
        f.writelines(lines)
        return [t_rel * 100, r_rel / np.pi * 180 * 100, ate, rpe_trans, rpe_rot * 180 / np.pi, trans_rmse, gt_dist,
                pred_dist, drift]

    def eval(self, gt_dir, result_dir, alignment=None, seqs=None, eval_seqs="", file_name_plot=''):
        """Evaluate sequences and return metrics."""
        self.gt_dir = gt_dir
        error_dir = os.path.join(result_dir, "errors")
        self.plot_path_dir = os.path.join(result_dir, "plot_path")
        self.plot_error_dir = os.path.join(result_dir, "plot_error")
        result_txt = os.path.join(result_dir, "result.txt")
        os.makedirs(error_dir, exist_ok=True)
        os.makedirs(self.plot_path_dir, exist_ok=True)
        os.makedirs(self.plot_error_dir, exist_ok=True)
        with open(result_txt, 'w') as f:
            result_file = os.path.join(result_dir, f"{eval_seqs}.txt")
            gt_file = os.path.join(gt_dir, f"{eval_seqs[:2]}.txt")
            if not os.path.exists(result_file):
                print(f"Pose file {result_file} not found")
                return []
            try:
                poses_result = self.load_poses_from_txt(result_file)
                poses_gt = self.load_poses_from_txt(gt_file)
                if len(poses_result) < len(poses_gt):
                    print(f"Warning: {result_file} has {len(poses_result)} poses, ground truth has {len(poses_gt)}")
                idx_0 = sorted(poses_result.keys())[0]
                pred_0, gt_0 = poses_result[idx_0], poses_gt[idx_0]
                for cnt in poses_result:
                    poses_result[cnt] = np.linalg.inv(pred_0) @ poses_result[cnt]
                    poses_gt[cnt] = np.linalg.inv(gt_0) @ poses_gt[cnt]
                if alignment == "scale":
                    poses_result = self.scale_optimization(poses_gt, poses_result)
                elif alignment in ["scale_7dof", "7dof", "6dof"]:
                    xyz_gt = np.array([poses_gt[cnt][:3, 3] for cnt in poses_result]).T
                    xyz_result = np.array([poses_result[cnt][:3, 3] for cnt in poses_result]).T
                    r, t, scale = umeyama_alignment(xyz_result, xyz_gt, alignment != "6dof")
                    align_transformation = np.eye(4)
                    align_transformation[:3, :3], align_transformation[:3, 3] = r, t
                    for cnt in poses_result:
                        poses_result[cnt][:3, 3] *= scale
                        if alignment in ["7dof", "6dof"]:
                            poses_result[cnt] = align_transformation @ poses_result[cnt]
                seq_err = self.calc_sequence_errors(poses_gt, poses_result)
                self.save_sequence_errors(seq_err, os.path.join(error_dir, f"{eval_seqs}.txt"))
                t_rel, r_rel = self.compute_overall_err(seq_err)
                ate = self.compute_ATE(poses_gt, poses_result)
                trans_rmse = self.compute_translational_rmse(poses_gt, poses_result)
                rpe_trans, rpe_rot = self.compute_RPE(poses_gt, poses_result)
                gt_dist = self.compute_total_distance(poses_gt)
                pred_dist = self.compute_total_distance(poses_result)
                drift = self.compute_drift(poses_gt, poses_result)
                pos_result = np.array([poses_result[k][:3, 3] for k in sorted(poses_result.keys())])
                pos_gt = np.array([poses_gt[k][:3, 3] for k in sorted(poses_gt.keys()) if k in poses_result])
                self.plot_trajectory(poses_gt, poses_result, eval_seqs, pos_gt, pos_result, file_name_plot)
                self.plot_error(self.compute_segment_error(seq_err), eval_seqs)
                metrics = self.write_result(f, eval_seqs,
                                            [t_rel, r_rel, ate, rpe_trans, rpe_rot, trans_rmse, gt_dist, pred_dist,
                                             drift])
                return metrics
            except Exception as e:
                print(f"Error processing sequence {eval_seqs}: {e}")
                return []
        return []

    @staticmethod
    def save_sequence_errors(err, file_name):
        """Save sequence errors to file."""
        with open(file_name, 'w') as f:
            for item in err:
                f.write(" ".join(map(str, item)) + "\n")

    def plot_trajectory(self, poses_gt, poses_result, seq, pos_gt, pos_result, file_name_plot):
        """Plot ground truth and predicted trajectories."""
        fig = plt.figure(figsize=(10, 10))
        ax = plt.gca()
        ax.set_aspect('equal')
        for label, poses in [("Ground Truth", poses_gt), ("Predicted", poses_result)]:
            pos_xz = np.array([[pose[0, 3], pose[2, 3]] for pose in [poses[k] for k in sorted(poses.keys())]])
            plt.plot(pos_xz[:, 0], pos_xz[:, 1], label=label)
        plt.legend(loc="upper right", prop={'size': 20})
        plt.xlabel('x (m)', fontsize=20)
        plt.ylabel('z (m)', fontsize=20)
        plt.savefig(os.path.join(self.plot_path_dir, file_name_plot), bbox_inches='tight', pad_inches=0)
        plt.close()
        self.plot_3d_trajectories_interactive(
            pos_result,
            pos_gt,
            title=f"3D Trajectory Comparison for Sequence {seq}",
            file_name=os.path.join(self.plot_path_dir, file_name_plot.split('.')[0] + ".html")
        )

    def plot_error(self, avg_segment_errs, seq):
        """Plot translation and rotation errors per segment length."""
        fontsize = 10
        plot_x = self.lengths
        plot_y = [avg_segment_errs[length][0] * 100 if avg_segment_errs[length] else 0 for length in self.lengths]
        plt.figure(figsize=(5, 5))
        plt.plot(plot_x, plot_y, "bs-", label="t_rel (%)")
        plt.xlabel('Path Length (m)', fontsize=fontsize)
        plt.ylabel('Translation Error (%)', fontsize=fontsize)
        plt.legend(loc="upper right", prop={'size': fontsize})
        plt.savefig(os.path.join(self.plot_error_dir, f"trans_err_{seq}.png"), bbox_inches='tight', pad_inches=0)
        plt.close()
        plot_y = [avg_segment_errs[length][1] / np.pi * 180 * 100 if avg_segment_errs[length] else 0 for length in
                  self.lengths]
        plt.figure(figsize=(5, 5))
        plt.plot(plot_x, plot_y, "bs-", label="r_rel (deg/100m)")
        plt.xlabel('Path Length (m)', fontsize=fontsize)
        plt.ylabel('Rotation Error (deg/100m)', fontsize=fontsize)
        plt.legend(loc="upper right", prop={'size': fontsize})
        plt.savefig(os.path.join(self.plot_error_dir, f"rot_err_{seq}.png"), bbox_inches='tight', pad_inches=0)
        plt.close()


def get_folders_in_dir(dir_path='./vo_data'):
    """Get a list of folders in directory."""
    return [os.path.join(dir_path, item) for item in os.listdir(dir_path) if
            os.path.isdir(os.path.join(dir_path, item))]


if __name__ == "__main__":
    dir_list = get_folders_in_dir('./result')
    print("Detected result folders:", dir_list)
    eval_headers = ["Method", "Alignment", "t_rel (%)", "r_rel (deg/100m)", "ATE (m)", "Trans RMSE (m)",
                    "RPE trans (m)", "RPE rot (deg)", "GT Dist (m)", "Pred Dist (m)", "Drift (m)"]
    comp_headers = ["Method", "Alignment", "Total Time (s)", "Frames Processed", "FPS", "Avg CPU (%)",
                    "Avg CPU Overall (%)", "Avg GPU (%)", "Avg GPU Mem (%)", "Avg GPU Power (W)", "Avg RAM (GB)"]
    results_table = []
    comp_table = []
    alignments = ['Direct']
    base_seqs = [0, 9]  # Base sequence numbers

    parser = argparse.ArgumentParser(description='KITTI VO evaluation with repetitions and computational metrics')
    parser.add_argument('--result', type=str, default='./result', help='Root directory of result folders')
    parser.add_argument('--gt_dir', type=str, default='dataset/kitti_odom/gt_poses/',
                        help='Ground truth poses directory')
    parser.add_argument('--seqs', nargs="+", type=int, default=base_seqs,
                        help='List of base sequence numbers to evaluate')
    parser.add_argument('--output_dir', type=str, default='plots', help='Directory for output plots and Excel')
    args = parser.parse_args([])

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    # Expand sequences to include repetitions (e.g., 00_1, 00_2, 00_3)
    seq_repeats = [(f"{seq:02}", i) for seq in args.seqs for i in range(1, 4)]
    print("Sequence repetitions:", seq_repeats)

    for folder in dir_list:
        for align in alignments:
            for seq, repeat in seq_repeats:
                seq_str = f"{seq}_{repeat}"
                eval_path = os.path.join(folder.replace('result', 'eval_matrix'), f"{seq_str}.txt")
                comp_path = os.path.join(folder.replace('result', 'Computational_matrix'), f"{seq_str}.txt")
                if not os.path.exists(eval_path) or not os.path.exists(comp_path):
                    print(f"Missing files for {seq_str} in {folder}")
                    continue
                parser_inner = argparse.ArgumentParser(description='KITTI VO evaluation')
                parser_inner.add_argument('--result', type=str, default=eval_path)
                parser_inner.add_argument('--align', type=str, default=align)
                parser_inner.add_argument('--seqs', nargs="+", type=int, default=[int(seq)])
                args_inner = parser_inner.parse_args([])
                eval_tool = KittiEvalOdom()
                try:
                    eval_metrics = eval_tool.eval(
                        args.gt_dir,
                        os.path.dirname(eval_path),
                        alignment=args_inner.align,
                        seqs=args_inner.seqs,
                        eval_seqs=seq_str,
                        file_name_plot=f"{align}_seq_{seq_str}_3D_Plot.png"
                    )
                    comp_metrics = parse_computational_metrics(comp_path)
                    if eval_metrics:
                        results_table.append([
                                                 os.path.basename(folder),
                                                 args_inner.align,
                                                 f"{eval_metrics[0]:.3f}",
                                                 f"{eval_metrics[1]:.3f}",
                                                 f"{eval_metrics[2]:.3f}",
                                                 f"{eval_metrics[5]:.3f}",
                                                 f"{eval_metrics[3]:.3f}",
                                                 f"{eval_metrics[4]:.3f}",
                                                 f"{eval_metrics[6]:.3f}",
                                                 f"{eval_metrics[7]:.3f}",
                                                 f"{eval_metrics[8]:.3f}"
                                             ] + [seq_str])
                    if comp_metrics['frames_processed'] > 0:
                        comp_table.append([
                                              os.path.basename(folder),
                                              align,
                                              f"{comp_metrics['total_time']:.3f}",
                                              f"{comp_metrics['frames_processed']}",
                                              f"{comp_metrics['fps']:.3f}",
                                              f"{comp_metrics['avg_cpu']:.3f}",
                                              f"{comp_metrics['avg_cpu_overall']:.3f}",
                                              f"{comp_metrics['avg_gpu']:.3f}",
                                              f"{comp_metrics['avg_gpu_mem']:.3f}",
                                              f"{comp_metrics['avg_gpu_power']:.3f}",
                                              f"{comp_metrics['avg_ram']:.3f}"
                                          ] + [seq_str])
                except Exception as e:
                    print(f"Error processing {folder} with alignment {align} for seq {seq_str}: {e}")

    # Display detailed results for each repetition
    print("\nDetailed Evaluation Results for Each Repetition:")
    print(tabulate(results_table, headers=eval_headers + ["Sequence"], tablefmt="grid"))
    df_detailed_eval = pd.DataFrame(results_table, columns=eval_headers + ["Sequence"])
    excel_detailed_eval_file = output_dir / "vo_evaluation_results_detailed_eval.xlsx"
    df_detailed_eval.to_excel(excel_detailed_eval_file, index=False)
    print(f"\nDetailed evaluation results saved to: {excel_detailed_eval_file}")

    print("\nDetailed Computational Results for Each Repetition:")
    print(tabulate(comp_table, headers=comp_headers + ["Sequence"], tablefmt="grid"))
    df_detailed_comp = pd.DataFrame(comp_table, columns=comp_headers + ["Sequence"])
    excel_detailed_comp_file = output_dir / "vo_evaluation_results_detailed_comp.xlsx"
    df_detailed_comp.to_excel(excel_detailed_comp_file, index=False)
    print(f"\nDetailed computational results saved to: {excel_detailed_comp_file}")

    # Aggregate results by base sequence and alignment
    averaged_eval_results = []
    averaged_comp_results = []
    for folder in set(row[0] for row in results_table):
        for align in alignments:
            for seq in args.seqs:
                seq_str = f"{seq:02}"
                eval_data = [row for row in results_table if
                             row[0] == folder and row[1] == align and row[11].startswith(seq_str + "_")]
                comp_data = [row for row in comp_table if
                             row[0] == folder and row[1] == align and row[11].startswith(seq_str + "_")]
                if eval_data and comp_data:
                    expected_repeats = [f"{seq_str}_{i}" for i in range(1, 4)]
                    eval_repeats = [row[11] for row in eval_data]
                    comp_repeats = [row[11] for row in comp_data]
                    if all(rep in eval_repeats for rep in expected_repeats) and all(
                            rep in comp_repeats for rep in expected_repeats):
                        eval_metrics_list = [[float(row[i]) for i in range(2, 11)] for row in eval_data]
                        comp_metrics_list = [[float(row[i]) for i in range(2, 11)] for row in comp_data]
                        avg_eval_metrics = [sum(col) / 3 for col in zip(*eval_metrics_list)]
                        avg_comp_metrics = [sum(col) / 3 for col in zip(*comp_metrics_list)]
                        averaged_eval_results.append([
                            folder, align,
                            f"{avg_eval_metrics[0]:.3f}", f"{avg_eval_metrics[1]:.3f}", f"{avg_eval_metrics[2]:.3f}",
                            f"{avg_eval_metrics[3]:.3f}", f"{avg_eval_metrics[4]:.3f}", f"{avg_eval_metrics[5]:.3f}",
                            f"{avg_eval_metrics[6]:.3f}", f"{avg_eval_metrics[7]:.3f}", f"{avg_eval_metrics[8]:.3f}",
                            seq_str
                        ])
                        averaged_comp_results.append([
                            folder, align,
                            f"{avg_comp_metrics[0]:.3f}", f"{avg_comp_metrics[1]}", f"{avg_comp_metrics[2]:.3f}",
                            f"{avg_comp_metrics[3]:.3f}", f"{avg_comp_metrics[4]:.3f}", f"{avg_comp_metrics[5]:.3f}",
                            f"{avg_comp_metrics[6]:.3f}", f"{avg_comp_metrics[7]:.3f}", f"{avg_comp_metrics[8]:.3f}",
                            seq_str
                        ])
                    else:
                        print(f"Warning: Incomplete repetitions for seq {seq_str} in {folder} with {align}")

    print("\nAveraged Evaluation Results:")
    print(tabulate(averaged_eval_results, headers=eval_headers + ["Base Sequence"], tablefmt="grid"))
    df_averaged_eval = pd.DataFrame(averaged_eval_results, columns=eval_headers + ["Base Sequence"])
    excel_averaged_eval_file = output_dir / "vo_evaluation_results_averaged_eval.xlsx"
    df_averaged_eval.to_excel(excel_averaged_eval_file, index=False)
    print(f"\nAveraged evaluation results saved to: {excel_averaged_eval_file}")

    print("\nAveraged Computational Results:")
    print(tabulate(averaged_comp_results, headers=comp_headers + ["Base Sequence"], tablefmt="grid"))
    df_averaged_comp = pd.DataFrame(averaged_comp_results, columns=comp_headers + ["Base Sequence"])
    excel_averaged_comp_file = output_dir / "vo_evaluation_results_averaged_comp.xlsx"
    df_averaged_comp.to_excel(excel_averaged_comp_file, index=False)
    print(f"\nAveraged computational results saved to: {excel_averaged_comp_file}")

    # Plotting (simplified, focusing on evaluation metrics for now)
    methods = sorted(set(row[0] for row in averaged_eval_results))
    labels = [f"{m} ({a})" for m in methods for a in alignments]
    eval_data = {metric: [] for metric in eval_headers[2:]}
    for m in methods:
        for a in alignments:
            for row in averaged_eval_results:
                if row[0] == m and row[1] == a:
                    for i, metric in enumerate(eval_headers[2:], 2):
                        eval_data[metric].append(float(row[i]))
                    break
            else:
                for metric in eval_headers[2:]:
                    eval_data[metric].append(0)

    figsize = (20, 8)
    bar_width = 0.1
    index = np.arange(len(labels))

    plt.figure(figsize=figsize)
    plt.bar(index - bar_width / 2, eval_data["t_rel (%)"], bar_width, label="t_rel (%)", color="dodgerblue")
    plt.bar(index + bar_width / 2, eval_data["r_rel (deg/100m)"], bar_width, label="r_rel (deg/100m)", color="tomato")
    plt.xlabel("Method (Alignment)")
    plt.ylabel("Error")
    plt.title("Averaged Relative Translation and Rotation Errors")
    plt.xticks(index, labels, rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "averaged_relative_errors.png")
    plt.close()

    print(f"Plots saved in '{args.output_dir}' directory.")
