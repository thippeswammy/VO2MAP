import os

folder = r'F:\RunningProjects\VisualOdemetry\Visual-odometry-tutorial\dump\DeepMatchVO\kitti_data\raw\09\image_1'
output_txt = r'image_paths.txt'

file_list = [os.path.join(folder, f) for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]

with open(output_txt, 'w') as f:
    for file in file_list:
        f.write(file + '\n')
