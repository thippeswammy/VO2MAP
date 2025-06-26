# Copyright (C) Huangying Zhan 2019. All rights reserved.

import argparse

from kitti_odometry import KittiEvalOdom

parser = argparse.ArgumentParser(description='KITTI evaluation')
parser.add_argument('--result', type=str, default='./result/example_1',
                    help="Result directory")
parser.add_argument('--align', type=str,
                    choices=['scale', 'scale_7dof', '7dof', '6dof', 'None'],
                    default='None',
                    help="alignment type")
parser.add_argument('--seqs',
                    nargs="+",
                    type=int,
                    help="sequences to be evaluated",
                    default=9)
args = parser.parse_args()

eval_tool = KittiEvalOdom()
gt_dir = "dataset/kitti_odom/gt_poses/"
result_dir = args.result

# continue_flag = input("Evaluate result in {}? [y/n]".format(result_dir))
# if continue_flag == "y":
print('args.align', args.align)
eval_tool.eval(
    gt_dir,
    result_dir,
    alignment=args.align,
    seqs=[args.seqs],
)
# else:
#     print("Double check the path!")

# python eval_odom.py --result result/example_6 --align 7dof --seqs 9
