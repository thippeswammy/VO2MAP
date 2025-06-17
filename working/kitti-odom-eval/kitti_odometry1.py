# Copyright (C) Huangying Zhan 2019. All rights reserved.

import argparse
import copy
import os
from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tabulate import tabulate


def scale_lse_solver(X, Y):
    """Least-square-error solver
    Compute optimal scaling factor so that s(X)-Y is minimum
    Args:
        X (KxN array): current data
        Y (KxN array): reference data
    Returns:
        scale (float): scaling factor
    """
    scale = np.sum(X * Y) / np.sum(X ** 2)
    return scale


def umeyama_alignment(x, y, with_scale=False):
    """
    Computes the least squares solution parameters of an Sim(m) matrix
    that minimizes the distance between a set of registered points.
    Umeyama, Shinji: Least-squares estimation of transformation parameters
                     between two point patterns. IEEE PAMI, 1991
    :param x: mxn matrix of points, m = dimension, n = nr. of data points
    :param y: mxn matrix of points, m = dimension, n = nr. of data points
    :param with_scale: set to True to align also the scale (default: 1.0 scale)
    :return: r, t, c - rotation matrix, translation vector and scale factor
    """
    if x.shape != y.shape:
        assert False, "x.shape not equal to y.shape"

    m, n = x.shape
    mean_x = x.mean(axis=1)
    mean_y = y.mean(axis=1)
    sigma_x = 1.0 / n * (np.linalg.norm(x - mean_x[:, np.newaxis]) ** 2)
    outer_sum = np.zeros((m, m))
    for i in range(n):
        outer_sum += np.outer((y[:, i] - mean_y), (x[:, i] - mean_x))
    cov_xy = np.multiply(1.0 / n, outer_sum)
    u, d, v = np.linalg.svd(cov_xy)
    s = np.eye(m)
    if np.linalg.det(u) * np.linalg.det(v) < 0.0:
        s[m - 1, m - 1] = -1
    r = u.dot(s).dot(v)
    c = 1 / sigma_x * np.trace(np.diag(d).dot(s)) if with_scale else 1.0
    t = mean_y - np.multiply(c, r.dot(mean_x))
    return r, t, c


class KittiEvalOdom:
    """Evaluate odometry result"""

    def __init__(self):
        self.lengths = [100, 200, 300, 400, 500, 600, 700, 800]
        self.num_lengths = len(self.lengths)

    def load_poses_from_txt(self, file_name):
        """Load poses from txt (KITTI format)"""
        try:
            with open(file_name, 'r') as f:
                s = f.readlines()
            poses = {}
            for cnt, line in enumerate(s):
                try:
                    line_split = [float(i) for i in line.strip().split(" ") if i != ""]
                    withIdx = len(line_split) == 13
                    if len(line_split) not in [12, 13]:
                        raise ValueError(f"Invalid number of values in line {cnt + 1}: {line.strip()}")
                    P = np.eye(4)
                    for row in range(3):
                        for col in range(4):
                            P[row, col] = line_split[row * 4 + col + (1 if withIdx else 0)]
                    frame_idx = line_split[0] if withIdx else cnt
                    poses[frame_idx] = P
                except ValueError as e:
                    print(f"Error parsing line {cnt + 1} in {file_name}: {e}")
                    continue
            if not poses:
                raise ValueError(f"No valid poses found in {file_name}")
            return poses
        except Exception as e:
            raise Exception(f"Failed to load poses from {file_name}: {e}")

    def trajectory_distances(self, poses):
        """Compute distance for each pose w.r.t frame-0"""
        dist = [0]
        sort_frame_idx = sorted(poses.keys())
        for i in range(len(sort_frame_idx) - 1):
            cur_frame_idx = sort_frame_idx[i]
            next_frame_idx = sort_frame_idx[i + 1]
            P1 = poses[cur_frame_idx]
            P2 = poses[next_frame_idx]
            dx = P1[0, 3] - P2[0, 3]
            dy = P1[1, 3] - P2[1, 3]
            dz = P1[2, 3] - P2[2, 3]
            dist.append(dist[i] + np.sqrt(dx ** 2 + dy ** 2 + dz ** 2))
        return dist

    def rotation_error(self, pose_error):
        """Compute rotation error"""
        a = pose_error[0, 0]
        b = pose_error[1, 1]
        c = pose_error[2, 2]
        d = 0.5 * (a + b + c - 1.0)
        rot_error = np.arccos(max(min(d, 1.0), -1.0))
        return rot_error

    def translation_error(self, pose_error):
        """Compute translation error"""
        dx = pose_error[0, 3]
        dy = pose_error[1, 3]
        dz = pose_error[2, 3]
        trans_error = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        return trans_error

    def last_frame_from_segment_length(self, dist, first_frame, length):
        """Find frame (index) that away from the first_frame with the required distance"""
        for i in range(first_frame, len(dist), 1):
            if dist[i] > (dist[first_frame] + length):
                return i
        return -1

    def calc_sequence_errors(self, poses_gt, poses_result):
        """Calculate sequence error"""
        err = []
        dist = self.trajectory_distances(poses_gt)
        self.step_size = 10
        for first_frame in range(0, len(poses_gt), self.step_size):
            for i in range(self.num_lengths):
                len_ = self.lengths[i]
                last_frame = self.last_frame_from_segment_length(dist, first_frame, len_)
                if last_frame == -1 or not (last_frame in poses_result.keys()) or not (
                        first_frame in poses_result.keys()):
                    continue
                pose_delta_gt = np.dot(np.linalg.inv(poses_gt[first_frame]), poses_gt[last_frame])
                pose_delta_result = np.dot(np.linalg.inv(poses_result[first_frame]), poses_result[last_frame])
                pose_error = np.dot(np.linalg.inv(pose_delta_result), pose_delta_gt)
                r_err = self.rotation_error(pose_error)
                t_err = self.translation_error(pose_error)
                num_frames = last_frame - first_frame + 1.0
                speed = len_ / (0.1 * num_frames)
                err.append([first_frame, r_err / len_, t_err / len_, len_, speed])
        return err

    def save_sequence_errors(self, err, file_name):
        """Save sequence error"""
        with open(file_name, 'w') as fp:
            for i in err:
                line_to_write = " ".join([str(j) for j in i])
                fp.writelines(line_to_write + "\n")

    def compute_overall_err(self, seq_err):
        """Compute average translation & rotation errors"""
        t_err = 0
        r_err = 0
        seq_len = len(seq_err)
        if seq_len > 0:
            for item in seq_err:
                r_err += item[1]
                t_err += item[2]
            ave_t_err = t_err / seq_len
            ave_r_err = r_err / seq_len
            return ave_t_err, ave_r_err
        return 0, 0

    def plot_trajectory(self, poses_gt, poses_result, seq):
        """Plot trajectory for both GT and prediction"""
        plot_keys = ["Ground Truth", "Ours"]
        fontsize_ = 20
        poses_dict = {"Ground Truth": poses_gt, "Ours": poses_result}
        fig = plt.figure()
        ax = plt.gca()
        ax.set_aspect('equal')
        for key in plot_keys:
            pos_xz = []
            frame_idx_list = sorted(poses_dict["Ours"].keys())
            for frame_idx in frame_idx_list:
                pose = poses_dict[key][frame_idx]
                pos_xz.append([pose[0, 3], pose[2, 3]])
            pos_xz = np.asarray(pos_xz)
            plt.plot(pos_xz[:, 0], pos_xz[:, 1], label=key)
        plt.legend(loc="upper right", prop={'size': fontsize_})
        plt.xticks(fontsize=fontsize_)
        plt.yticks(fontsize=fontsize_)
        plt.xlabel('x (m)', fontsize=fontsize_)
        plt.ylabel('z (m)', fontsize=fontsize_)
        fig.set_size_inches(10, 10)
        png_title = f"sequence_{seq:02}"
        fig_pdf = os.path.join(self.plot_path_dir, f"{png_title}.png")
        plt.savefig(fig_pdf, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

    def plot_error(self, avg_segment_errs, seq):
        """Plot per-length error"""
        fontsize_ = 10
        # Translation error
        plot_x = []
        plot_y = []
        for len_ in self.lengths:
            plot_x.append(len_)
            plot_y.append(avg_segment_errs[len_][0] * 100 if len(avg_segment_errs[len_]) > 0 else 0)
        fig = plt.figure()
        plt.plot(plot_x, plot_y, "bs-", label="Translation Error")
        plt.ylabel('Translation Error (%)', fontsize=fontsize_)
        plt.xlabel('Path Length (m)', fontsize=fontsize_)
        plt.legend(loc="upper right", prop={'size': fontsize_})
        fig.set_size_inches(5, 5)
        fig_pdf = os.path.join(self.plot_error_dir, f"trans_err_{seq:02}.png")
        plt.savefig(fig_pdf, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        # Rotation error
        plot_y = []
        for len_ in self.lengths:
            plot_y.append(avg_segment_errs[len_][1] / np.pi * 180 * 100 if len(avg_segment_errs[len_]) > 0 else 0)
        fig = plt.figure()
        plt.plot(plot_x, plot_y, "bs-", label="Rotation Error")
        plt.ylabel('Rotation Error (deg/100m)', fontsize=fontsize_)
        plt.xlabel('Path Length (m)', fontsize=fontsize_)
        plt.legend(loc="upper right", prop={'size': fontsize_})
        fig.set_size_inches(5, 5)
        fig_pdf = os.path.join(self.plot_error_dir, f"rot_err_{seq:02}.png")
        plt.savefig(fig_pdf, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

    def compute_segment_error(self, seq_errs):
        """Calculate average errors for different segment"""
        segment_errs = {len_: [] for len_ in self.lengths}
        avg_segment_errs = {}
        for err in seq_errs:
            len_ = err[3]
            t_err = err[2]
            r_err = err[1]
            segment_errs[len_].append([t_err, r_err])
        for len_ in self.lengths:
            if segment_errs[len_]:
                avg_t_err = np.mean(np.asarray(segment_errs[len_])[:, 0])
                avg_r_err = np.mean(np.asarray(segment_errs[len_])[:, 1])
                avg_segment_errs[len_] = [avg_t_err, avg_r_err]
            else:
                avg_segment_errs[len_] = []
        return avg_segment_errs

    def compute_ATE(self, gt, pred):
        """Compute RMSE of ATE"""
        errors = []
        idx_0 = list(pred.keys())[0]
        gt_0 = gt[idx_0]
        pred_0 = pred[idx_0]
        for i in pred:
            cur_gt = gt[i]
            gt_xyz = cur_gt[:3, 3]
            cur_pred = pred[i]
            pred_xyz = cur_pred[:3, 3]
            align_err = gt_xyz - pred_xyz
            errors.append(np.sqrt(np.sum(align_err ** 2)))
        ate = np.sqrt(np.mean(np.asarray(errors) ** 2))
        return ate

    def compute_RPE(self, gt, pred):
        """Compute RPE"""
        trans_errors = []
        rot_errors = []
        for i in list(pred.keys())[:-1]:
            gt1 = gt[i]
            gt2 = gt[i + 1]
            gt_rel = np.linalg.inv(gt1) @ gt2
            pred1 = pred[i]
            pred2 = pred[i + 1]
            pred_rel = np.linalg.inv(pred1) @ pred2
            rel_err = np.linalg.inv(gt_rel) @ pred_rel
            trans_errors.append(self.translation_error(rel_err))
            rot_errors.append(self.rotation_error(rel_err))
        rpe_trans = np.mean(np.asarray(trans_errors))
        rpe_rot = np.mean(np.asarray(rot_errors))
        return rpe_trans, rpe_rot

    def scale_optimization(self, gt, pred):
        """Optimize scaling factor"""
        pred_updated = copy.deepcopy(pred)
        xyz_pred = []
        xyz_ref = []
        for i in pred:
            pose_pred = pred[i]
            pose_ref = gt[i]
            xyz_pred.append(pose_pred[:3, 3])
            xyz_ref.append(pose_ref[:3, 3])
        xyz_pred = np.asarray(xyz_pred)
        xyz_ref = np.asarray(xyz_ref)
        scale = scale_lse_solver(xyz_pred, xyz_ref)
        for i in pred_updated:
            pred_updated[i][:3, 3] *= scale
        return pred_updated

    def compute_total_distance(self, poses):
        """Compute total distance traveled"""
        keys = sorted(poses.keys())
        dist = 0.0
        for i in range(1, len(keys)):
            p1 = poses[keys[i - 1]][:3, 3]
            p2 = poses[keys[i]][:3, 3]
            dist += np.linalg.norm(p2 - p1)
        return dist

    def compute_drift(self, gt_poses, pred_poses):
        """Compute drift as Euclidean distance between final GT and predicted poses"""
        keys = sorted(gt_poses.keys())
        final_gt = gt_poses[keys[-1]][:3, 3]
        final_pred = pred_poses[keys[-1]][:3, 3]
        return np.linalg.norm(final_gt - final_pred)

    def write_result(self, f, seq, errs):
        """Write result into a txt file"""
        ave_t_err, ave_r_err, ate, rpe_trans, rpe_rot, gt_dist, pred_dist, drift = errs
        lines = []
        list_ = []
        lines.append(f"Sequence: \t {seq} \n")
        list_.append(ave_t_err * 100)
        lines.append(f"Trans. err. (%): \t {ave_t_err * 100:.3f} \n")
        list_.append(ave_r_err / np.pi * 180 * 100)
        lines.append(f"Rot. err. (deg/100m): \t {ave_r_err / np.pi * 180 * 100:.3f} \n")
        list_.append(ate)
        lines.append(f"ATE (m): \t {ate:.3f} \n")
        list_.append(rpe_trans)
        lines.append(f"RPE (m): \t {rpe_trans:.3f} \n")
        list_.append(rpe_rot * 180 / np.pi)
        lines.append(f"RPE (deg): \t {rpe_rot * 180 / np.pi:.3f} \n")
        list_.append(gt_dist)
        lines.append(f"GT Distance (m): \t {gt_dist:.3f} \n")
        list_.append(pred_dist)
        lines.append(f"Predicted Distance (m): \t {pred_dist:.3f} \n")
        list_.append(drift)
        lines.append(f"Final Drift (m): \t {drift:.3f} \n\n")
        for line in lines:
            f.writelines(line)
        return list_

    def eval(self, gt_dir, result_dir, alignment=None, seqs=None):
        """Evaluate required/available sequences"""
        seq_list = ["{:02}".format(i) for i in range(0, 11)]
        self.gt_dir = gt_dir
        ave_t_errs = []
        ave_r_errs = []
        seq_ate = []
        seq_rpe_trans = []
        seq_rpe_rot = []
        gt_dists = []
        pred_dists = []
        drifts = []
        error_dir = os.path.join(result_dir, "errors")
        self.plot_path_dir = os.path.join(result_dir, "plot_path")
        self.plot_error_dir = os.path.join(result_dir, "plot_error")
        result_txt = os.path.join(result_dir, "result.txt")
        with open(result_txt, 'w') as f:
            if not os.path.exists(error_dir):
                os.makedirs(error_dir)
            if not os.path.exists(self.plot_path_dir):
                os.makedirs(self.plot_path_dir)
            if not os.path.exists(self.plot_error_dir):
                os.makedirs(self.plot_error_dir)
            if seqs is None:
                available_seqs = sorted(glob(os.path.join(result_dir, "*.txt")))
                self.eval_seqs = [int(i[-6:-4]) for i in available_seqs if i[-6:-4] in seq_list]
            else:
                self.eval_seqs = seqs
            for i in self.eval_seqs:
                self.cur_seq = f'{i:02}'
                file_name = f'{i:02}.txt'
                result_file = os.path.join(result_dir, file_name)
                gt_file = os.path.join(self.gt_dir, file_name)
                if not os.path.exists(result_file):
                    print(f"Pose file {result_file} not found")
                    continue
                try:
                    poses_result = self.load_poses_from_txt(result_file)
                    poses_gt = self.load_poses_from_txt(gt_file)
                    self.result_file_name = result_file
                    # Check for frame count mismatch
                    if len(poses_result) < len(poses_gt):
                        print(
                            f"Warning: {result_file} has {len(poses_result)} poses, but ground truth has {len(poses_gt)}")
                    idx_0 = sorted(list(poses_result.keys()))[0]
                    pred_0 = poses_result[idx_0]
                    gt_0 = poses_gt[idx_0]
                    for cnt in poses_result:
                        poses_result[cnt] = np.linalg.inv(pred_0) @ poses_result[cnt]
                        poses_gt[cnt] = np.linalg.inv(gt_0) @ poses_gt[cnt]
                    if alignment == "scale":
                        poses_result = self.scale_optimization(poses_gt, poses_result)
                    elif alignment in ["scale_7dof", "7dof", "6dof"]:
                        xyz_gt = []
                        xyz_result = []
                        for cnt in poses_result:
                            xyz_gt.append([poses_gt[cnt][0, 3], poses_gt[cnt][1, 3], poses_gt[cnt][2, 3]])
                            xyz_result.append(
                                [poses_result[cnt][0, 3], poses_result[cnt][1, 3], poses_result[cnt][2, 3]])
                        xyz_gt = np.asarray(xyz_gt).transpose(1, 0)
                        xyz_result = np.asarray(xyz_result).transpose(1, 0)
                        r, t, scale = umeyama_alignment(xyz_result, xyz_gt, alignment != "6dof")
                        align_transformation = np.eye(4)
                        align_transformation[:3, :3] = r
                        align_transformation[:3, 3] = t
                        for cnt in poses_result:
                            poses_result[cnt][:3, 3] *= scale
                            if alignment in ["7dof", "6dof"]:
                                poses_result[cnt] = align_transformation @ poses_result[cnt]
                    seq_err = self.calc_sequence_errors(poses_gt, poses_result)
                    self.save_sequence_errors(seq_err, os.path.join(error_dir, file_name))
                    avg_segment_errs = self.compute_segment_error(seq_err)
                    ave_t_err, ave_r_err = self.compute_overall_err(seq_err)
                    # print(f"Sequence: {i}")
                    # print(f"Translational error (%): {ave_t_err * 100:.3f}")
                    # print(f"Rotational error (deg/100m): {ave_r_err / np.pi * 180 * 100:.3f}")
                    ave_t_errs.append(ave_t_err)
                    ave_r_errs.append(ave_r_err)
                    ate = self.compute_ATE(poses_gt, poses_result)
                    seq_ate.append(ate)
                    # print(f"ATE (m): {ate:.3f}")
                    rpe_trans, rpe_rot = self.compute_RPE(poses_gt, poses_result)
                    seq_rpe_trans.append(rpe_trans)
                    seq_rpe_rot.append(rpe_rot)
                    # print(f"RPE (m): {rpe_trans:.3f}")
                    # print(f"RPE (deg): {rpe_rot * 180 / np.pi:.3f}")
                    gt_dist = self.compute_total_distance(poses_gt)
                    pred_dist = self.compute_total_distance(poses_result)
                    gt_dists.append(gt_dist)
                    pred_dists.append(pred_dist)
                    # print(f"GT Distance (m): {gt_dist:.3f}")
                    # print(f"Predicted Distance (m): {pred_dist:.3f}")
                    drift = self.compute_drift(poses_gt, poses_result)
                    drifts.append(drift)
                    # print(f"Final Drift (m): {drift:.3f}")
                    self.plot_trajectory(poses_gt, poses_result, i)
                    self.plot_error(avg_segment_errs, i)
                    list_ = self.write_result(f, i, [ave_t_err, ave_r_err, ate, rpe_trans, rpe_rot, gt_dist, pred_dist,
                                                     drift])
                except Exception as e:
                    print(f"Error processing sequence {i} in {result_dir} with alignment {alignment}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            return list_
            # print("-------------------- For Copying ------------------------------")
            # for i in range(len(ave_t_errs)):
            # print(f"{ave_t_errs[i] * 100:.2f}")
            # print(f"{ave_r_errs[i] / np.pi * 180 * 100:.2f}")
            # print(f"{seq_ate[i]:.2f}")
            # print(f"{seq_rpe_trans[i]:.3f}")
            # print(f"{seq_rpe_rot[i] * 180 / np.pi:.3f}")
            # print(f"{gt_dists[i]:.3f}")
            # print(f"{pred_dists[i]:.3f}")
            # print(f"{drifts[i]:.3f}")


def get_folders_in_dir(dir_path='./vo_data'):
    """Returns a list of all folders in the specified directory."""
    return [os.path.join(dir_path, item) for item in os.listdir(dir_path)
            if os.path.isdir(os.path.join(dir_path, item))]


if __name__ == "__main__":
    dir_list = get_folders_in_dir()
    print("Detected result folders:", dir_list)
    headers = ["Result Folder", "Alignment", "Trans. err. (%)", "Rot. err. (deg/100m)", "ATE (m)", "RPE (m)",
               "RPE (deg)", "GT Dist (m)", "Pred Dist (m)", "Drift (m)"]
    results_table = []
    for file in dir_list:
        for align in ['scale', 'scale_7dof', '7dof', '6dof']:
            parser = argparse.ArgumentParser(description='KITTI evaluation')
            parser.add_argument('--result', type=str, default=file)
            parser.add_argument('--align', type=str, choices=['scale', 'scale_7dof', '7dof', '6dof'], default=align)
            parser.add_argument('--seqs', nargs="+", type=int, default=[9])
            args = parser.parse_args([])
            eval_tool = KittiEvalOdom()
            gt_dir = "dataset/kitti_odom/gt_poses/"
            result_dir = args.result
            try:
                metrics = eval_tool.eval(
                    gt_dir,
                    result_dir,
                    alignment=args.align,
                    seqs=args.seqs,
                )
                results_table.append([
                    os.path.basename(result_dir),
                    args.align,
                    f"{metrics[0]:.3f}",
                    f"{metrics[1]:.3f}",
                    f"{metrics[2]:.3f}",
                    f"{metrics[3]:.3f}",
                    f"{metrics[4]:.3f}",
                    f"{metrics[5]:.3f}",
                    f"{metrics[6]:.3f}",
                    f"{metrics[7]:.3f}"
                ])
            except Exception as e:
                print(f"Error while processing {file} with alignment {align}: {e}")
    print("\nResults Table:")
    print(tabulate(results_table, headers=headers, tablefmt="grid"))
    df = pd.DataFrame(results_table, columns=headers)

    # Save to Excel
    excel_filename = "./plots/vo_evaluation_results.xlsx"
    df.to_excel(excel_filename, index=False)

    print(f"\nResults saved to Excel: {excel_filename}")

    methods = sorted(set(row[0] for row in results_table))
    alignments = ['scale', 'scale_7dof', '7dof', '6dof']
    labels = [f"{m} ({a})" for m in methods for a in alignments]

    # Initialize data lists
    trans_err = []
    rot_err = []
    ate = []
    drift = []
    gt_dist = []
    pred_dist = []
    rpe_trans = []
    rpe_rot = []

    # Populate data lists based on results_table
    for m in methods:
        for a in alignments:
            for row in results_table:
                if row[0] == m and row[1] == a:
                    trans_err.append(float(row[2]))
                    rot_err.append(float(row[3]))
                    ate.append(float(row[4]))
                    rpe_trans.append(float(row[5]))
                    rpe_rot.append(float(row[6]))
                    gt_dist.append(float(row[7]))
                    pred_dist.append(float(row[8]))
                    drift.append(float(row[9]))
                    break
            else:
                # If no data for this method-alignment pair, append zeros
                trans_err.append(0)
                rot_err.append(0)
                ate.append(0)
                rpe_trans.append(0)
                rpe_rot.append(0)
                gt_dist.append(0)
                pred_dist.append(0)
                drift.append(0)

    # Plot settings
    # plt.style.use('seaborn')
    figsize = (20, 8)
    bar_width = 0.35
    index = np.arange(len(labels))

    # Create output directory
    output_dir = Path("plots")
    output_dir.mkdir(exist_ok=True)

    # Plot 1: Translational and Rotational Errors
    plt.figure(figsize=figsize)
    plt.bar(index - bar_width / 2, trans_err, bar_width, label="Trans. err. (%)", color="dodgerblue")
    plt.bar(index + bar_width / 2, rot_err, bar_width, label="Rot. err. (deg/100m)", color="tomato")
    plt.xlabel("Method (Alignment)")
    plt.ylabel("Error")
    plt.title("Translational and Rotational Errors (Sequence 9)")
    plt.xticks(index, labels, rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "trans_rot_errors.png")
    plt.close()

    # Plot 2: ATE and Drift
    plt.figure(figsize=figsize)
    plt.bar(index - bar_width / 2, ate, bar_width, label="ATE (m)", color="teal")
    plt.bar(index + bar_width / 2, drift, bar_width, label="Drift (m)", color="purple")
    plt.xlabel("Method (Alignment)")
    plt.ylabel("Error (m)")
    plt.title("ATE and Drift (Sequence 9)")
    plt.xticks(index, labels, rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "ate_drift.png")
    plt.close()

    # Plot 3: Predicted vs. GT Distance
    plt.figure(figsize=figsize)
    plt.bar(index - bar_width / 2, gt_dist, bar_width, label="GT Distance (m)", color="dodgerblue")
    plt.bar(index + bar_width / 2, pred_dist, bar_width, label="Predicted Distance (m)", color="teal")
    plt.xlabel("Method (Alignment)")
    plt.ylabel("Distance (m)")
    plt.title("Predicted vs. GT Distance (Sequence 9)")
    plt.xticks(index, labels, rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "distance_comparison.png")
    plt.close()

    # Plot 4: RPE (Translation and Rotation)
    plt.figure(figsize=figsize)
    plt.bar(index - bar_width / 2, rpe_trans, bar_width, label="RPE (m)", color="orange")
    plt.bar(index + bar_width / 2, rpe_rot, bar_width, label="RPE (deg)", color="purple")
    plt.xlabel("Method (Alignment)")
    plt.ylabel("RPE")
    plt.title("RPE Translation and Rotation (Sequence 9)")
    plt.xticks(index, labels, rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "rpe_comparison.png")
    plt.close()

    print("Plots saved in 'plots' directory.")
