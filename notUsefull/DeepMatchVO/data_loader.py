from __future__ import division

import os

import numpy as np
import tensorflow as tf

tf.compat.v1.disable_eager_execution()
expected_height = 128
expected_width = 416


class DataLoader(object):
    def __init__(self,
                 dataset_dir=None,
                 batch_size=None,
                 img_height=None,
                 img_width=None,
                 num_source=None,
                 num_scales=None,
                 read_pose=False,
                 match_num=0):
        self.dataset_dir = dataset_dir
        self.batch_size = batch_size
        self.img_height = img_height
        self.img_width = img_width
        self.num_source = num_source
        self.num_scales = num_scales
        self.read_pose = read_pose
        self.match_num = match_num

    def load_train_batch(self):
        """
        Load a batch of training instances using the new tensorflow
        Dataset api.
        """

        def load_train_batch(self):
            def _parse_train_img(img_path):
                img_buffer = tf.io.read_file(img_path)
                image_decoded = tf.image.decode_jpeg(img_buffer)
                tgt_image, src_image_stack = self.unpack_image_sequence(
                    image_decoded, self.img_height, self.img_width, self.num_source)
                return tgt_image, src_image_stack

            def _batch_preprocessing(stack_images, intrinsics, optional_data):
                intrinsics = tf.cast(intrinsics, tf.float32)
                image_all = tf.concat([stack_images[0], stack_images[1]], axis=2)
                if self.match_num == 0:
                    image_all, intrinsics = self.data_augmentation(
                        image_all, intrinsics, self.img_height, self.img_width)
                tgt_image = image_all[:, :, :3]
                src_image_stack = image_all[:, :, 3:]
                intrinsics = self.get_multi_scale_intrinsics(intrinsics, self.num_scales)
                return tgt_image, src_image_stack, intrinsics, optional_data

            file_list = self.format_file_list(self.dataset_dir, 'train')
            self.steps_per_epoch = len(file_list['image_file_list']) // self.batch_size

            image_dataset = tf.data.Dataset.from_tensor_slices(file_list['image_file_list'])
            image_dataset = image_dataset.map(_parse_train_img, num_parallel_calls=4)

            cam_intrinsics = self.load_cam_intrinsics(file_list['cam_file_list'])
            intrinsics_dataset = tf.data.Dataset.from_tensor_slices(cam_intrinsics)

            if self.read_pose:
                all_poses = self.load_poses_6dof(file_list['cam_file_list'])
                pose_dataset = tf.data.Dataset.from_tensor_slices(all_poses)
                dataset = tf.data.Dataset.zip((image_dataset, intrinsics_dataset, pose_dataset))
            elif self.match_num > 0:
                all_matches = self.load_matches(file_list['cam_file_list'])
                match_dataset = tf.data.Dataset.from_tensor_slices(all_matches)
                dataset = tf.data.Dataset.zip((image_dataset, intrinsics_dataset, match_dataset))
            else:
                dataset = tf.data.Dataset.zip((image_dataset, intrinsics_dataset))

            dataset = dataset.batch(self.batch_size).prefetch(1)
            return dataset
        # def _parse_train_img(img_path):
        #     with tf.device('/cpu:0'):
        #         img_buffer = tf.read_file(img_path)
        #         image_decoded = tf.image.decode_jpeg(img_buffer)
        #         tgt_image, src_image_stack = \
        #             self.unpack_image_sequence(
        #                 image_decoded, self.img_height, self.img_width, self.num_source)
        #     return tgt_image, src_image_stack

        # def _batch_preprocessing(stack_images, intrinsics, optional_data):
        #     intrinsics = tf.cast(intrinsics, tf.float32)
        #     image_all = tf.concat([stack_images[0], stack_images[1]], axis=3)
        #
        #     if self.match_num == 0:  # otherwise matches coords are wrong
        #         image_all, intrinsics = self.data_augmentation(
        #             image_all, intrinsics, self.img_height, self.img_width)
        #     tgt_image = image_all[:, :, :, :3]
        #     src_image_stack = image_all[:, :, :, 3:]
        #     intrinsics = self.get_multi_scale_intrinsics(intrinsics, self.num_scales)
        #     return tgt_image, src_image_stack, intrinsics, optional_data

        # file_list = self.format_file_list(self.dataset_dir, 'train')
        # self.steps_per_epoch = int(len(file_list['image_file_list']) // self.batch_size)
        #
        # input_image_names_ph = tf.placeholder(tf.string, shape=[None], name='input_image_names_ph')
        # image_dataset = tf.data.Dataset.from_tensor_slices(
        #     input_image_names_ph).map(_parse_train_img)
        #
        # cam_intrinsics_ph = tf.placeholder(tf.float32, [None, 3, 3], name='cam_intrinsics_ph')
        # intrinsics_dataset = tf.data.Dataset.from_tensor_slices(cam_intrinsics_ph)

        # datasets = (image_dataset, intrinsics_dataset, intrinsics_dataset)
        # if self.read_pose:
        #     poses_ph = tf.placeholder(tf.float32, [None, self.num_source + 1, 6], name='poses_ph')
        #     pose_dataset = tf.data.Dataset.from_tensor_slices(poses_ph)
        #     datasets = (image_dataset, intrinsics_dataset, pose_dataset)
        # if self.match_num > 0:
        #     matches_ph = tf.placeholder(tf.float32, [None, self.num_source, self.match_num, 4], name='matches_ph')
        #     match_dataset = tf.data.Dataset.from_tensor_slices(matches_ph)
        #     datasets = (image_dataset, intrinsics_dataset, match_dataset)

        # all_dataset = tf.data.Dataset.zip(datasets)
        # all_dataset = all_dataset.batch(self.batch_size).repeat().prefetch(self.batch_size * 4)
        # all_dataset = all_dataset.map(_batch_preprocessing)
        # iterator = tf.compat.v1.data.make_initializable_iterator(all_dataset)
        # return iterator

    def load_test_batch(self, image_sequence_names):
        """load a batch of test images for inference"""

        # def _parse_test_img(img_path):
        #     with tf.device('/cpu:0'):
        #         print('img_path11=>', img_path, type(img_path))
        #         if not isinstance(img_path, tf.Tensor) or img_path.dtype != tf.string:
        #             img_path = tf.convert_to_tensor(str(img_path.numpy()), dtype=tf.string)
        #         img_path = tf.convert_to_tensor(img_path, dtype=tf.string)  # Force string type
        #         img_buffer = tf.io.read_file(img_path)
        #         # img_buffer = tf.read_file(img_path)
        #         print('img_path22=>', img_path, type(img_path))
        #         image_decoded = tf.image.decode_jpeg(img_buffer)
        #     return image_decoded
        #
        # print('image_sequence_names=>', image_sequence_names)
        # image_dataset = tf.data.Dataset.from_tensor_slices(image_sequence_names).map(
        #     _parse_test_img).batch(self.batch_size).prefetch(self.batch_size * 4)
        # iterator = image_dataset.make_initializable_iterator()
        # return iterator
        def _parse_test_img(img_path):
            img_buffer = tf.io.read_file(img_path)
            image_decoded = tf.image.decode_png(img_buffer, channels=1)  # Decode as grayscale
            image_decoded = tf.image.resize(image_decoded, [self.img_height, self.img_width * (
                        self.num_source + 1)])  # Resize to concatenated width
            image_decoded = tf.image.grayscale_to_rgb(image_decoded)  # Convert to RGB
            return image_decoded

        image_dataset = tf.data.Dataset.from_tensor_slices(image_sequence_names)
        image_dataset = image_dataset.map(_parse_test_img, num_parallel_calls=4)
        image_dataset = image_dataset.batch(self.batch_size).prefetch(1)
        return image_dataset

    def init_data_pipeline(self, sess, batch_sample):
        def _load_cam_intrinsics(cam_filelist):
            all_cam_intrinsics = []
            for filename in cam_filelist:
                with open(filename) as f:
                    line = f.readlines()
                    cam_intri_vec = [float(num) for num in line[0].split(',')]
                    cam_intrinsics = np.reshape(cam_intri_vec, [3, 3])
                    all_cam_intrinsics.append(cam_intrinsics)
            all_cam_intrinsics = np.stack(all_cam_intrinsics, axis=0)
            return all_cam_intrinsics

        def load_cam_intrinsics(self, cam_filelist):
            all_cam_intrinsics = []
            for filename in cam_filelist:
                with open(filename) as f:
                    line = f.readline()
                    cam_intri_vec = [float(num) for num in line.split(',')]
                    cam_intrinsics = np.reshape(cam_intri_vec, [3, 3])
                    all_cam_intrinsics.append(cam_intrinsics)
            return np.stack(all_cam_intrinsics, axis=0)

        def _load_poses_6dof(cam_filelist):
            all_poses = []
            for filename in cam_filelist:
                with open(filename) as f:
                    lines = f.readlines()
                    one_sample_pose = []
                    for i in range(1, len(lines)):
                        pose = [float(num) for num in lines[i].split(',')]
                        pose_vec = np.reshape(pose, [6])
                        one_sample_pose.append(pose_vec)
                    one_sample_pose = np.stack(one_sample_pose, axis=0)
                all_poses.append(one_sample_pose)
            all_poses = np.stack(all_poses, axis=0)
            return all_poses

        def load_poses_6dof(self, cam_filelist):
            all_poses = []
            for filename in cam_filelist:
                with open(filename) as f:
                    lines = f.readlines()
                    one_sample_pose = []
                    for line in lines[1:]:
                        pose = [float(num) for num in line.split(',')]
                        one_sample_pose.append(np.reshape(pose, [6]))
                    all_poses.append(np.stack(one_sample_pose, axis=0))
            return np.stack(all_poses, axis=0)

        def _load_matches(cam_file_list):
            all_matches = []
            for filename in cam_file_list:
                with open(filename) as f:
                    lines = f.readlines()
                    # read num_source * match_num (x,y) pairs
                    image_matches = []
                    for i in range(self.num_source):
                        one_matches = []
                        for j in range(self.match_num):
                            match_coords = [float(num) for num in lines[1 + i * self.match_num + j].split(',')]
                            match_vec = np.reshape(match_coords, [4])
                            one_matches.append(match_vec)
                        one_matches = np.stack(one_matches, axis=0)
                        image_matches.append(one_matches)
                    image_matches = np.stack(image_matches, axis=0)
                    all_matches.append(image_matches)
            all_matches = np.stack(all_matches, axis=0)
            return all_matches

        def load_matches(self, cam_file_list):
            all_matches = []
            for filename in cam_file_list:
                with open(filename) as f:
                    lines = f.readlines()
                    image_matches = []
                    for i in range(self.num_source):
                        one_matches = []
                        for j in range(self.match_num):
                            match_coords = [float(num) for num in lines[1 + i * self.match_num + j].split(',')]
                            one_matches.append(np.reshape(match_coords, [4]))
                        image_matches.append(np.stack(one_matches, axis=0))
                    all_matches.append(np.stack(image_matches, axis=0))
            return np.stack(all_matches, axis=0)

        # load cam_intrinsics using native python
        file_list = self.format_file_list(self.dataset_dir, 'train')
        print('load camera intrinsics...')
        cam_intrinsics = _load_cam_intrinsics(file_list['cam_file_list'])

        input_dict = {'data_loading/input_image_names_ph:0':
                          file_list['image_file_list'][:self.batch_size *
                                                        self.steps_per_epoch],
                      'data_loading/cam_intrinsics_ph:0':
                          cam_intrinsics[:self.batch_size * self.steps_per_epoch]}
        if self.read_pose:
            print('load pose data...')
            all_poses = _load_poses_6dof(file_list['cam_file_list'])
            input_dict['data_loading/poses_ph:0'] = all_poses[:self.batch_size * self.steps_per_epoch]
        if self.match_num > 0:
            print('load matches data...')
            all_matches = _load_matches(file_list['cam_file_list'])
            input_dict['data_loading/matches_ph:0'] = all_matches
        # sess.run(iterator.initializer)
        sess.run(batch_sample.initializer, feed_dict=input_dict)

    # def make_intrinsics_matrix(self, fx, fy, cx, cy):
    #     # Assumes batch input
    #     batch_size = tf.shape(fx)[0]
    #     zeros = tf.zeros_like(fx)
    #     r1 = tf.stack([fx, zeros, cx], axis=1)
    #     r2 = tf.stack([zeros, fy, cy], axis=1)
    #     r3 = tf.constant([0., 0., 1.], shape=[1, 3])
    #     r3 = tf.tile(r3, [batch_size, 1])
    #     intrinsics = tf.stack([r1, r2, r3], axis=1)
    #     return intrinsics
    def make_intrinsics_matrix(self, fx, fy, cx, cy):
        batch_size = tf.shape(fx)[0]
        zeros = tf.zeros_like(fx)
        r1 = tf.stack([fx, zeros, cx], axis=1)
        r2 = tf.stack([zeros, fy, cy], axis=1)
        r3 = tf.tile(tf.constant([[0., 0., 1.]]), [batch_size, 1])
        intrinsics = tf.stack([r1, r2, r3], axis=1)
        return intrinsics

    def data_augmentation(self, im, intrinsics, out_h, out_w):
        # Random scaling
        def random_scaling(im, intrinsics):
            in_h, in_w = tf.shape(im)[0], tf.shape(im)[1]
            scaling = tf.random.uniform([2], 1.0, 1.15)
            x_scaling, y_scaling = scaling[0], scaling[1]
            out_h_ = tf.cast(tf.cast(in_h, tf.float32) * y_scaling, tf.int32)
            out_w_ = tf.cast(tf.cast(in_w, tf.float32) * x_scaling, tf.int32)
            im = tf.image.resize(im, [out_h_, out_w_])
            fx = intrinsics[:, 0, 0] * x_scaling
            fy = intrinsics[:, 1, 1] * y_scaling
            cx = intrinsics[:, 0, 2] * x_scaling
            cy = intrinsics[:, 1, 2] * y_scaling
            intrinsics = self.make_intrinsics_matrix(fx, fy, cx, cy)
            return im, intrinsics

        # Random cropping
        def random_cropping(im, intrinsics, out_h, out_w):
            in_h, in_w = tf.shape(im)[0], tf.shape(im)[1]
            offset_y = tf.random.uniform([], 0, in_h - out_h + 1, dtype=tf.int32)
            offset_x = tf.random.uniform([], 0, in_w - out_w + 1, dtype=tf.int32)
            im = tf.image.crop_to_bounding_box(im, offset_y, offset_x, out_h, out_w)
            fx = intrinsics[:, 0, 0]
            fy = intrinsics[:, 1, 1]
            cx = intrinsics[:, 0, 2] - tf.cast(offset_x, tf.float32)
            cy = intrinsics[:, 1, 2] - tf.cast(offset_y, tf.float32)
            intrinsics = self.make_intrinsics_matrix(fx, fy, cx, cy)
            return im, intrinsics

        im, intrinsics = random_scaling(im, intrinsics)
        im, intrinsics = random_cropping(im, intrinsics, out_h, out_w)
        im = tf.cast(im, dtype=tf.uint8)
        return im, intrinsics

    # def format_file_list(self, data_root, split):
    #     all_list = {}
    #     with open(data_root + '/%s.txt' % split, 'r') as f:
    #         frames = f.readlines()
    #     subfolders = [x.split(' ')[0] for x in frames]
    #     frame_ids = [x.split(' ')[1][:-1] for x in frames]
    #     image_file_list = [os.path.join(data_root, subfolders[i],
    #                                     frame_ids[i] + '.jpg') for i in range(len(frames))]
    #     cam_file_list = [os.path.join(data_root, subfolders[i],
    #                                   frame_ids[i] + '_cam.txt') for i in range(len(frames))]
    #     all_list['image_file_list'] = image_file_list
    #     all_list['cam_file_list'] = cam_file_list
    #     return all_list
    def format_file_list(self, data_root, split):
        with open(os.path.join(data_root, f'{split}.txt'), 'r') as f:
            frames = f.readlines()
        subfolders = [x.split(' ')[0] for x in frames]
        frame_ids = [x.split(' ')[1].strip() for x in frames]
        image_file_list = [os.path.join(data_root, subfolders[i], f'{frame_ids[i]}.png') for i in range(len(frames))]
        cam_file_list = [os.path.join(data_root, subfolders[i], f'{frame_ids[i]}_cam.txt') for i in range(len(frames))]
        return {'image_file_list': image_file_list, 'cam_file_list': cam_file_list}

    # def unpack_image_sequence(self, image_seq, img_height, img_width, num_source):
    #     # Assuming the center image is the target frame
    #     tgt_start_idx = int(img_width * (num_source // 2))
    #     tgt_image = tf.slice(image_seq,
    #                          [0, tgt_start_idx, 0],
    #                          [-1, img_width, -1])
    #     # Source frames before the target frame
    #     src_image_1 = tf.slice(image_seq,
    #                            [0, 0, 0],
    #                            [-1, int(img_width * (num_source // 2)), -1])
    #     # Source frames after the target frame
    #     src_image_2 = tf.slice(image_seq,
    #                            [0, int(tgt_start_idx + img_width), 0],
    #                            [-1, int(img_width * (num_source // 2)), -1])
    #     src_image_seq = tf.concat([src_image_1, src_image_2], axis=1)
    #     # Stack source frames along the color channels (i.e. [H, W, N*3])
    #     src_image_stack = tf.concat([tf.slice(src_image_seq,
    #                                           [0, i * img_width, 0],
    #                                           [-1, img_width, -1])
    #                                  for i in range(num_source)], axis=2)
    #     src_image_stack.set_shape([img_height, img_width, num_source * 3])
    #     tgt_image.set_shape([img_height, img_width, 3])
    #     return tgt_image, src_image_stack
    def unpack_image_sequence(self, image_seq, img_height, img_width, num_source):
        tgt_start_idx = int(img_width * (num_source // 2))
        tgt_image = tf.slice(image_seq, [0, tgt_start_idx, 0], [-1, img_width, -1])
        src_image_1 = tf.slice(image_seq, [0, 0, 0], [-1, tgt_start_idx, -1])
        src_image_2 = tf.slice(image_seq, [0, tgt_start_idx + img_width, 0], [-1, tgt_start_idx, -1])
        src_image_seq = tf.concat([src_image_1, src_image_2], axis=1)
        src_image_stack = tf.concat(
            [tf.slice(src_image_seq, [0, i * img_width, 0], [-1, img_width, -1])
             for i in range(num_source)], axis=2)
        src_image_stack.set_shape([img_height, img_width, num_source * 3])
        tgt_image.set_shape([img_height, img_width, 3])
        return tgt_image, src_image_stack

    def batch_unpack_image_sequence(self, image_seq, img_height, img_width, num_source):
        # Ensure input image_seq is resized to the expected dimensions
        image_seq = tf.image.resize(image_seq, [img_height, img_width * (num_source + 1)])
        image_seq.set_shape([None, img_height, img_width * (num_source + 1), 3])  # Expect RGB

        # Assuming the center image is the target frame
        tgt_start_idx = int(img_width * (num_source // 2))
        tgt_image = tf.slice(image_seq, [0, 0, tgt_start_idx, 0], [-1, img_height, img_width, 3])

        # Source frames before and after the target frame
        src_image_1 = tf.slice(image_seq, [0, 0, 0, 0], [-1, img_height, tgt_start_idx, 3])
        src_image_2 = tf.slice(image_seq, [0, 0, tgt_start_idx + img_width, 0],
                               [-1, img_height, img_width * (num_source // 2), 3])
        src_image_seq = tf.concat([src_image_1, src_image_2], axis=2)

        # Stack source frames along the channel dimension
        src_image_stack = tf.concat(
            [tf.slice(src_image_seq, [0, 0, i * img_width, 0], [-1, img_height, img_width, 3])
             for i in range(num_source)], axis=3)

        # Set shapes explicitly
        tgt_image.set_shape([None, img_height, img_width, 3])
        src_image_stack.set_shape([None, img_height, img_width, num_source * 3])

        print('tgt_image.shape', tgt_image.shape)
        print('src_image_stack.shape', src_image_stack.shape)
        return tgt_image, src_image_stack

    # def get_multi_scale_intrinsics(self, intrinsics, num_scales):
    #     intrinsics_mscale = []
    #     # Scale the intrinsics accordingly for each scale
    #     for s in range(num_scales):
    #         fx = intrinsics[:, 0, 0] / (2 ** s)
    #         fy = intrinsics[:, 1, 1] / (2 ** s)
    #         cx = intrinsics[:, 0, 2] / (2 ** s)
    #         cy = intrinsics[:, 1, 2] / (2 ** s)
    #         intrinsics_mscale.append(
    #             self.make_intrinsics_matrix(fx, fy, cx, cy))
    #     intrinsics_mscale = tf.stack(intrinsics_mscale, axis=1)
    #     return intrinsics_mscale

    def get_multi_scale_intrinsics(self, intrinsics, num_scales):
        intrinsics_mscale = []
        for s in range(num_scales):
            fx = intrinsics[:, 0, 0] / (2 ** s)
            fy = intrinsics[:, 1, 1] / (2 ** s)
            cx = intrinsics[:, 0, 2] / (2 ** s)
            cy = intrinsics[:, 1, 2] / (2 ** s)
            intrinsics_mscale.append(self.make_intrinsics_matrix(fx, fy, cx, cy))
        return tf.stack(intrinsics_mscale, axis=1)
