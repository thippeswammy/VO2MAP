from __future__ import division

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from glob import glob

import numpy as np
import tensorflow as tf
from common_utils import complete_batch_size
from data_loader import DataLoader
from deep_slam import DeepSlam
from kitti_eval.pose_evaluation_utils import dump_pose_seq_TUM

flags = tf.app.flags
flags.DEFINE_integer("batch_size", 1, "The size of of a sample batch")
flags.DEFINE_integer("img_height", 128, "Image height")
flags.DEFINE_integer("img_width", 416, "Image width")
flags.DEFINE_integer("seq_length", 5, "Sequence length for each example")
flags.DEFINE_integer("test_seq", 9, "Sequence id to test")
flags.DEFINE_string("dataset_dir", None, "Raw odometry dataset directory")
flags.DEFINE_string("concat_img_dir", None, "Preprocess image dataset directory")
flags.DEFINE_string("output_dir", None, "Output directory")
flags.DEFINE_string("ckpt_file", None, "checkpoint file")
FLAGS = flags.FLAGS
'''
--test_seq 09 --dataset_dir data/kitti_raw --output_dir output/09 --ckpt_file checkpoints/depth_model/model-258000 --seq_length 3 --concat_img_dir kitti_data/raw
'''


def load_kitti_image_sequence_names(dataset_dir, frames, seq_length):
    image_sequence_names = []
    target_inds = []
    frame_num = len(frames)
    for tgt_idx in range(frame_num):
        # if not is_valid_sample(frames, tgt_idx, FLAGS.seq_length):
        #     continue
        curr_drive, curr_frame_id = frames[tgt_idx].split(' ')
        img_filename = os.path.join(dataset_dir, '%s/%s/%s.png' % (curr_drive, 'image_0', curr_frame_id))
        image_sequence_names.append(img_filename)
        target_inds.append(tgt_idx)
    print('len((image_sequence_names))', len((image_sequence_names)))
    return image_sequence_names, target_inds


# def main():
#     # get input images
#     if not os.path.isdir(FLAGS.output_dir):
#         os.makedirs(FLAGS.output_dir)
#     concat_img_dir = os.path.join(FLAGS.concat_img_dir, '%.2d' % FLAGS.test_seq)
#     max_src_offset = int((FLAGS.seq_length - 1) / 2)
#     N = len(glob(concat_img_dir + '/*.jpg')) + 2 * max_src_offset
#     test_frames = ['%.2d %.6d' % (FLAGS.test_seq, n) for n in range(N)]
#     print("files_name2=>", FLAGS.dataset_dir + 'sequences/%.2d/times.txt' % FLAGS.test_seq)
#     with open(FLAGS.dataset_dir + 'sequences/%.2d/times.txt' % FLAGS.test_seq, 'r') as f:
#         times = f.readlines()
#     times = np.array([float(s[:-1]) for s in times])
#     print(times[:10])
#     with tf.Session() as sess:
#         # setup input tensor
#         loader = DataLoader(FLAGS.concat_img_dir, FLAGS.batch_size, FLAGS.img_height, FLAGS.img_width,
#                             FLAGS.seq_length - 1)
#         image_sequence_names, tgt_inds = load_kitti_image_sequence_names(FLAGS.concat_img_dir, test_frames,
#                                                                          FLAGS.seq_length)
#         image_sequence_names = complete_batch_size(image_sequence_names, FLAGS.batch_size)
#         tgt_inds = complete_batch_size(tgt_inds, FLAGS.batch_size)
#         assert len(tgt_inds) == len(image_sequence_names)
#         print('image_sequence_names=>', image_sequence_names)
#         batch_sample = loader.load_test_batch(image_sequence_names)
#         # sess.run(batch_sample.initializer)
#
#         iterator = tf.compat.v1.data.make_initializable_iterator(batch_sample)
#         sess.run(iterator.initializer)
#         next_element = iterator.get_next()
#         # input_batch = batch_sample.get_next()
#         iterator = tf.compat.v1.data.make_initializable_iterator(batch_sample)
#         input_batch = iterator.get_next()
#
#         input_batch.set_shape([FLAGS.batch_size, FLAGS.img_height, FLAGS.img_width * FLAGS.seq_length, 3])
#
#         # init system
#         system = DeepSlam()
#         system.setup_inference(FLAGS.img_height, FLAGS.img_width,
#                                'pose', FLAGS.seq_length, FLAGS.batch_size, input_batch)
#         saver = tf.train.Saver([var for var in tf.trainable_variables()])
#         saver.restore(sess, FLAGS.ckpt_file)
#
#         round_num = len(image_sequence_names) // FLAGS.batch_size
#         for i in range(round_num):
#             pred = system.inference(sess, mode='pose')
#             for j in range(FLAGS.batch_size):
#                 tgt_idx = tgt_inds[i * FLAGS.batch_size + j]
#                 pred_poses = pred['pose'][j]
#                 # Insert the target pose [0, 0, 0, 0, 0, 0] to the middle
#                 pred_poses = np.insert(pred_poses, max_src_offset, np.zeros((1, 6)), axis=0)
#                 curr_times = times[tgt_idx - max_src_offset: tgt_idx + max_src_offset + 1]
#                 out_file = FLAGS.output_dir + '%.6d.txt' % (tgt_idx - max_src_offset)
#                 dump_pose_seq_TUM(out_file, pred_poses, curr_times)

def main():
    # get input images
    if not os.path.isdir(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)

    concat_img_dir = os.path.join(FLAGS.concat_img_dir, '%.2d' % FLAGS.test_seq)
    max_src_offset = int((FLAGS.seq_length - 1) / 2)
    N = len(glob(concat_img_dir + '/*.png')) + 2 * max_src_offset
    # test_frames = ['%.2d %.6d' % (FLAGS.test_seq, n) for n in range(N)]
    valid_frames = []
    for i in range(1591):
        valid_frames.append('%.2d %.6d' % (FLAGS.test_seq, i))
    test_frames = valid_frames
    print("files_name=>", FLAGS.dataset_dir + 'sequences/%.2d/times.txt' % FLAGS.test_seq)
    with open(FLAGS.dataset_dir + 'sequences/%.2d/times.txt' % FLAGS.test_seq, 'r') as f:
        times = f.readlines()
    times = np.array([float(s.strip()) for s in times])
    print(times[:10])

    with tf.compat.v1.Session() as sess:
        # setup input tensor
        loader = DataLoader(
            FLAGS.concat_img_dir,
            FLAGS.batch_size,
            FLAGS.img_height,
            FLAGS.img_width,
            FLAGS.seq_length - 1
        )

        image_sequence_names, tgt_inds = load_kitti_image_sequence_names(
            FLAGS.concat_img_dir,
            test_frames,
            FLAGS.seq_length
        )
        image_sequence_names = complete_batch_size(image_sequence_names, FLAGS.batch_size)
        tgt_inds = complete_batch_size(tgt_inds, FLAGS.batch_size)
        assert len(tgt_inds) == len(image_sequence_names)

        print('image_sequence_names=>', image_sequence_names)

        batch_sample = loader.load_test_batch(image_sequence_names)

        #  Correct: create iterator once
        iterator = tf.compat.v1.data.make_initializable_iterator(batch_sample)
        input_batch = iterator.get_next()

        # Set the expected input shape
        # In main() function, around line 146
        input_batch.set_shape([FLAGS.batch_size, FLAGS.img_height, FLAGS.img_width * FLAGS.seq_length, 3])
        # # In main() function, around line 146
        # input_batch.set_shape([FLAGS.batch_size, FLAGS.img_height, FLAGS.img_width * FLAGS.seq_length, 3])

        # Initialize all required variables and the iterator
        sess.run(tf.compat.v1.global_variables_initializer())
        sess.run(tf.compat.v1.local_variables_initializer())
        sess.run(iterator.initializer)

        # init system
        system = DeepSlam()
        system.setup_inference(
            FLAGS.img_height,
            FLAGS.img_width,
            'pose',
            FLAGS.seq_length,
            FLAGS.batch_size,
            input_batch
        )

        saver = tf.compat.v1.train.Saver([var for var in tf.compat.v1.trainable_variables()])
        saver.restore(sess, FLAGS.ckpt_file)

        round_num = len(image_sequence_names) // FLAGS.batch_size
        for i in range(round_num):
            pred = system.inference(sess, mode='pose')
            for j in range(FLAGS.batch_size):
                tgt_idx = tgt_inds[i * FLAGS.batch_size + j]
                pred_poses = pred['pose'][j]

                # Insert the target pose [0, 0, 0, 0, 0, 0] to the middle
                pred_poses = np.insert(pred_poses, max_src_offset, np.zeros((1, 6)), axis=0)

                curr_times = times[tgt_idx - max_src_offset: tgt_idx + max_src_offset + 1]
                out_file = os.path.join(FLAGS.output_dir, '%.6d.txt' % (tgt_idx - max_src_offset))
                dump_pose_seq_TUM(out_file, pred_poses, curr_times)


if __name__ == '__main__':
    main()
'''
test_kitti_pose.py --test_seq 09 --dataset_dir F:\RunningProjects\VisualOdemetry\Visual-odometry-tutorial\dump\DeepMatchVO\data\kitti_raw --output_dir F:\RunningProjects\VisualOdemetry\Visual-odometry-tutorial\dump\DeepMatchVO\output\09 --ckpt_file F:\RunningProjects\VisualOdemetry\Visual-odometry-tutorial\dump\DeepMatchVO\checkpoints\depth_model\model-258000 --seq_length 3 --concat_img_dir F:\RunningProjects\VisualOdemetry\Visual-odometry-tutorial\dump\DeepMatchVO\kitti_data\raw
'''
