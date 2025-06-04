# import os
#
# folder = r'F:\RunningProjects\VisualOdemetry\Visual-odometry-tutorial\dump\DeepMatchVO\kitti_data\raw\09\image_1'
# output_txt = r'image_paths.txt'
#
# file_list = [os.path.join(folder, f) for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
#
# with open(output_txt, 'w') as f:
#     for file in file_list:
#         f.write(file + '\n')

# import cv2
#
# img = cv2.imread(
#     "F:/RunningProjects/VisualOdemetry/Visual-odometry-tutorial/dump/DeepMatchVO/kitti_data/raw/09/image_0/000000.png",
#     cv2.IMREAD_GRAYSCALE)
# print(img.shape)  # Should print something like (128, 1248)

import tensorflow as tf

checkpoint_path = 'checkpoints/depth_model/model-258000'
reader = tf.train.load_checkpoint(checkpoint_path)
var_to_shape_map = reader.get_variable_to_shape_map()
for key in var_to_shape_map:
    print(f"{key}: {var_to_shape_map[key]}")
