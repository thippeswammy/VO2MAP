import argparse
import os

from kitti_odometry import KittiEvalOdom


def get_folders_in_dir(dir_path='./vo_data'):
    """
    Returns a list of all folders in the specified directory.
    """
    return [os.path.join(dir_path, item) for item in os.listdir(dir_path)
            if os.path.isdir(os.path.join(dir_path, item))]


dir_list = get_folders_in_dir()
print("Detected result folders:", dir_list)

# Header for table
results_table = []
headers = ["Result Folder", "Alignment", "Trans. err. (%)", "Rot. err. (deg/100m)", "ATE (m)", "RPE (m)", "RPE (deg)"]

for file in dir_list:
    for align in ['scale', 'scale_7dof', '7dof', '6dof']:
        parser = argparse.ArgumentParser(description='KITTI evaluation')
        parser.add_argument('--result', type=str, default=file)
        parser.add_argument('--align', type=str, choices=['scale', 'scale_7dof', '7dof', '6dof'], default=align)
        parser.add_argument('--seqs', nargs="+", type=int, default=[9])
        args = parser.parse_args([])  # Empty list to avoid reading sys.argv

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

            # Append one row to the results table
            results_table.append([
                os.path.basename(result_dir),
                args.align,
                f"{metrics[0]:.3f}",
                f"{metrics[1]:.3f}",
                f"{metrics[2]:.3f}",
                f"{metrics[3]:.3f}",
                f"{metrics[4]:.3f}"
            ])

        except Exception as e:
            print(f"Error while processing {file} with alignment {align}: {e}")

# Display the table using tabulate (optional)
try:
    from tabulate import tabulate

    print(tabulate(results_table, headers=headers, tablefmt="grid"))
except ImportError:
    # If tabulate is not installed, fallback to plain print
    print("\nResults Table:")
    print(" | ".join(headers))
    for row in results_table:
        print(" | ".join(row))
