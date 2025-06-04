# Set paths (adjust according to your folder locations)
kitti_dir = "F:\RunningProjects\VisualOdemetry\Visual-odometry-tutorial\dump\DeepMatchVO\kitti_data"
kitti_raw_dir = f"{kitti_dir}/raw"
kitti_raw_odom = f"{kitti_dir}/odometry"
make3d_dir = "F:/Datasets/make3d"
kitti_raw_dump_dir = f"{kitti_dir}/raw_dump"
kitti_eigen_test_dir = f"{kitti_dir}/eigen_test"
kitti_odom = f"{kitti_dir}/kitti_odom"
kitti_odom5 = f"{kitti_dir}/kitti_odom5"
kitti_odom_match3 = f"{kitti_dir}/kitti_odom_match3"
kitti_odom_match5 = f"{kitti_dir}/kitti_odom_match5"
cityscapes_dir = "F:/Datasets/cityscapes"
cityscapes_dump = f"{cityscapes_dir}/dump"
output_folder = "./output"
model_idx = 258000
save_freq_step = 4000
checkpoint_dir = "./ckpt"
match_num = 100
depth_pred_file = f"{output_folder}/model-{model_idx}.npy"
#
## 1. Generate training/testing data
#import os
#
## Prepare KITTI odometry dataset
#os.system(
#    f"python data/prepare_train_data.py --dataset_dir={kitti_raw_odom} --dataset_name=kitti_odom "
#    f"--dump_root={kitti_odom_match3} --seq_length=3 --img_width=416 --img_height=128 "
#    f"--num_threads=8 --generate_test True"
#)
#
## Prepare KITTI raw dataset (Eigen split)
#os.system(
#    f"python data/prepare_train_data.py --dataset_dir={kitti_raw_dir} --dataset_name=kitti_raw_eigen "
#    f"--dump_root={kitti_raw_dump_dir} --seq_length=3 --img_width=416 --img_height=128 "
#    f"--num_threads=8 --match_num={match_num}"
#)
#
## 2. Train on KITTI odometry dataset
#os.system(
#    f"python train.py --dataset_dir={kitti_odom_match3} --checkpoint_dir={checkpoint_dir} "
#    f"--img_width=416 --img_height=128 --batch_size=4 --seq_length 3 "
#    f"--max_steps 300000 --save_freq 2000 --learning_rate 0.001 --num_scales 1 "
#    f"--init_ckpt_file={checkpoint_dir}/model-{model_idx} --continue_train=True --match_num {match_num}"
#)
#
## 3. Train on KITTI Eigen split
#os.system(
#    f"python train.py --dataset_dir={kitti_raw_dump_dir} --checkpoint_dir={checkpoint_dir} "
#    f"--img_width=416 --img_height=128 --batch_size=4 --seq_length 3 "
#    f"--max_steps 300000 --save_freq {save_freq_step} --learning_rate 0.001 --num_scales 1 "
#    f"--match_num {match_num} --init_ckpt_file={checkpoint_dir}/model-{model_idx} --continue_train=True"
#)

# 4. Test depth model
r = 250000
depth_ckpt_file = f"{checkpoint_dir}/model-{r}"
depth_pred_file = f"{output_folder}/model-{r}.npy"

os.system(
    f"python test_kitti_depth.py --dataset_dir {kitti_raw_dir} --output_dir {output_folder} --ckpt_file {depth_ckpt_file}"
)
os.system(
    f"python kitti_eval/eval_depth.py --kitti_dir={kitti_raw_dir} --pred_file {depth_pred_file}"
)
#
## 5. Test pose model
#sl = 3
#r = 258000
#pose_ckpt_file = f"{checkpoint_dir}/model-{r}"
#
#for seq_num in ["09", "10"]:
#    out_seq_path = os.path.join(output_folder, seq_num)
#    os.system(f"rmdir /S /Q {out_seq_path}")  # delete folder (Windows CMD-style)
#    print(f"seq {seq_num}")
#    os.system(
#        f"python test_kitti_pose.py --test_seq {seq_num} --dataset_dir {kitti_raw_odom} "
#        f"--output_dir {out_seq_path}/ --ckpt_file {pose_ckpt_file} --seq_length {sl} "
#        f"--concat_img_dir {kitti_odom_match3}"
#    )
#    os.system(
#        f"python kitti_eval/eval_pose.py --gtruth_dir=kitti_eval/pose_data/ground_truth/seq{sl}/{seq_num}/ "
#        f"--pred_dir={out_seq_path}/"
#    )
