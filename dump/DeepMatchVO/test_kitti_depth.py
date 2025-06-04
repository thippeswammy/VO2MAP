from __future__ import division

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import PIL.Image as pil
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc
import tensorflow as tf

tf.compat.v1.disable_eager_execution()
config = tf.compat.v1.ConfigProto(device_count={'GPU': 0})  # disable GPU
session = tf.compat.v1.Session(config=config)
from deep_slam import DeepSlam

flags = tf.app.flags
flags.DEFINE_integer("batch_size", 1, "The size of of a sample batch")
flags.DEFINE_integer("img_height", 128, "Image height")
flags.DEFINE_integer("img_width", 416, "Image width")
flags.DEFINE_string("dataset_dir", None, "Dataset directory")
flags.DEFINE_string("output_dir", None, "Output directory")
flags.DEFINE_string("ckpt_file", None, "checkpoint file")
flags.DEFINE_boolean("show", False, "checkpoint file")
FLAGS = flags.FLAGS


def get_downsample_images(files, img_height, img_width, write_image):
    lr_img_files = []
    for file in files:
        dump_img_file = os.path.splitext(file)[0] + '_lr.jpg'
        lr_img_files.append(dump_img_file)
        if write_image:
            img = scipy.misc.imread(file)
            img = scipy.misc.imresize(img, (img_height, img_width))
            scipy.misc.imsave(dump_img_file, img.astype(np.uint8))
            print('write', dump_img_file)
    return lr_img_files


def main(_):
    with open('data/kitti/test_files_eigen.txt', 'r') as f:
        test_files = f.readlines()
        # print("test_files==>", test_files)
        # print("FLAGS.dataset_dir===>", FLAGS.dataset_dir)
        test_files = [t[:-1] for t in test_files]
    # print("test_files=>", test_files)
    if not os.path.exists(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)
    basename = os.path.basename(FLAGS.ckpt_file)
    system = DeepSlam()
    system.setup_inference(img_height=FLAGS.img_height,
                           img_width=FLAGS.img_width,
                           batch_size=FLAGS.batch_size,
                           mode='depth')
    saver = tf.train.Saver([var for var in tf.model_variables()])

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        saver.restore(sess, FLAGS.ckpt_file)
        pred_all = []
        # for t in range(0, len(test_files), FLAGS.batch_size):
        for t in range(0, 201, FLAGS.batch_size):
            if t % 100 == 0:
                print('processing %s: %d/%d' % (basename, t, len(test_files)))
            inputs = np.zeros(
                (FLAGS.batch_size, FLAGS.img_height, FLAGS.img_width, 3),
                dtype=np.uint8)
            for b in range(FLAGS.batch_size):
                idx = t + b
                if idx >= len(test_files):
                    break
                # fh = open(test_files[idx], 'r')
                file_path = test_files[idx]
                # print('file_path==>', file_path)
                # if not os.path.icsabs(file_path):
                #     file_path = os.path.join(args.dataset_dir, file_path)
                fh = open(file_path, 'rb')

                # raw_im = pil.open(fh)
                raw_im = pil.open(fh).convert('RGB')
                # changed
                # scaled_im = raw_im.resize((FLAGS.img_width, FLAGS.img_height), pil.ANTIALIAS)
                scaled_im = raw_im.resize((FLAGS.img_width, FLAGS.img_height), Image.Resampling.LANCZOS)

                inputs[b] = np.array(scaled_im)
            pred = system.inference(sess, 'depth', inputs)
            for b in range(FLAGS.batch_size):
                idx = t + b
                if idx >= len(test_files):
                    break
                tmp_depth = pred['depth'][b, :, :, 0]
                pred_all.append(tmp_depth)

                # obtain scaled image and depth image
                fh = open(test_files[idx], 'r')
                # changes
                # raw_im = pil.open(fh)
                with open(file_path, 'rb') as fh:
                    raw_im = Image.open(fh)
                    raw_im = raw_im.convert('RGB')
                scaled_im = raw_im.resize((FLAGS.img_width, FLAGS.img_height), pil.ANTIALIAS)
                scaled_im = np.array(scaled_im)
                depth_img = np.squeeze(pred['depth'][b, :, :, 0])

                # show the image side by side
                if FLAGS.show:
                    plt.figure()
                    plt.subplot(211)
                    plt.imshow(scaled_im)

                    plt.subplot(212)
                    plt.imshow(depth_img, cmap='gray')
                    plt.show()

        output_file = FLAGS.output_dir + '/' + basename
        np.save(output_file, pred_all)
        print('Save predicted depth map to', output_file)


if __name__ == '__main__':
    tf.app.run()
