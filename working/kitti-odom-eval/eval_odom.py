# Copyright (C) Huangying Zhan 2019. All rights reserved.

import argparse

from kitti_odometry import KittiEvalOdom

parser = argparse.ArgumentParser(description='KITTI evaluation')
parser.add_argument('--result', type=str, required=True,
                    help="Result directory")
parser.add_argument('--align', type=str,
                    choices=['scale', 'scale_7dof', '7dof', '6dof','None'],
                    default='None',
                    help="alignment type")
parser.add_argument('--seqs',
                    nargs="+",
                    type=int,
                    help="sequences to be evaluated",
                    default=None)
args = parser.parse_args()

eval_tool = KittiEvalOdom()
gt_dir = "dataset/kitti_odom/gt_poses/"
result_dir = args.result

# continue_flag = input("Evaluate result in {}? [y/n]".format(result_dir))
# if continue_flag == "y":
print('args.align',args.align)
eval_tool.eval(
    gt_dir,
    result_dir,
    alignment=args.align,
    seqs=args.seqs,
)
# else:
#     print("Double check the path!")

# python eval_odom.py --result result/example_6 --align 7dof --seqs 9

'''
------------------------------------------------------------
                        example_
Translational error (%):  105.23726159710118
Rotational error (deg/100m):  31.388224933249237
ATE (m):  47.84657161465758
RPE (m):  2.0689149305537256
RPE (deg):  1.3937203513975827



------------------------------------------------------------
                    example_1(vo_opticalFlow) -> 6dof
Translational error (%):  10.300693515038747
Rotational error (deg/100m):  3.213037491320276
ATE (m):  35.13270848325184
RPE (m):  0.030242469586663406
RPE (deg):  0.28702441717664073
------------------------------------------------------------
                    example_1(vo_opticalFlow) -> 7dof (auto scale)
Translational error (%):  10.692465937038829
Rotational error (deg/100m):  3.213037491320276
ATE (m):  33.626748402608676
RPE (m):  0.0564386652101945
RPE (deg):  0.28702441717664073

'''
