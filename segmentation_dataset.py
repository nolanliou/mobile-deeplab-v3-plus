"""PASCAL VOC2012 dataset
"""
import os
import collections
import tensorflow as tf

import matplotlib.pyplot as plt

import input_preprocess
# borrow from tensorflow models

_ITEMS_TO_DESCRIPTIONS = {
    'image': 'A color image of varying height and width.',
    'labels_class': ('A semantic segmentation label whose size matches image.'
                     'Its values range from 0 (background) to num_classes.'),
}

# Named tuple to describe the dataset properties.
DatasetDescriptor = collections.namedtuple(
    'DatasetDescriptor',
    ['subset_to_sizes',   # Subset of the dataset into training, val, and test.
     'num_classes',   # Number of semantic classes, including the background
                      # class (if exists). For example, there are 20
                      # foreground classes + 1 background class in the PASCAL
                      # VOC 2012 dataset. Thus, we set num_classes=21.
     'ignore_label',  # Ignore label value.
    ]
)

_CITYSCAPES_INFORMATION = DatasetDescriptor(
    subset_to_sizes={
        'train': 2975,
        'val': 500,
    },
    num_classes=19,
    ignore_label=255,
)

_PASCAL_VOC_2012_INFORMATION = DatasetDescriptor(
    subset_to_sizes={
        'train': 1464,
        'train_aug': 10582,
        'trainval': 2913,
        'val': 1449,
    },
    num_classes=21,
    ignore_label=255,
)

# These number (i.e., 'train'/'test') seems to have to be hard coded
# You are required to figure it out for your training/testing example.
_ADE20K_INFORMATION = DatasetDescriptor(
    subset_to_sizes={
        'train': 20210,  # num of samples in images/training
        'val': 2000,  # num of samples in images/validation
    },
    num_classes=151,
    ignore_label=0,
)


_DATASETS_INFORMATION = {
    'cityscapes': _CITYSCAPES_INFORMATION,
    'pascal_voc2012': _PASCAL_VOC_2012_INFORMATION,
    'ade20k': _ADE20K_INFORMATION,
}

# Default file pattern of TFRecord of TensorFlow Example.
_FILE_PATTERN = '%s-*'



class SegmentationDataset(object):
    """Segmentation Dataset

    """

    def __init__(self,
                 dataset_name,
                 dataset_dir,
                 subset,
                 model_input_height,
                 model_input_width,
                 min_scale_factor=1.,
                 max_scale_factor=1.,
                 scale_factor_step_size=0,
                 is_training=True):
        self.dataset_name = dataset_name
        self.dataset_dir = dataset_dir
        self.subset = subset
        self.model_input_height = model_input_height
        self.model_input_width = model_input_width
        self.min_scale_factor = min_scale_factor
        self.max_scale_factor = max_scale_factor
        self.scale_factor_step_size = scale_factor_step_size
        self.is_training = is_training

    def get_num_classes(self):
        return _DATASETS_INFORMATION[self.dataset_name].num_classes

    def _get_filenames(self):
        if self.dataset_name not in _DATASETS_INFORMATION:
            raise ValueError('The specified dataset is not supported yet.')

        subset_to_sizes = _DATASETS_INFORMATION[self.dataset_name].\
            subset_to_sizes

        if self.subset not in subset_to_sizes:
            raise ValueError('data subset name %s not recognized' % self.subset)

        file_pattern = _FILE_PATTERN
        data_source = os.path.join(self.dataset_dir,
                                   file_pattern % self.subset)
        if '*' in data_source or '?' in data_source or '[' in data_source:
            data_files = tf.gfile.Glob(data_source)
        else:
            data_files = [data_source]
        if not data_files:
            raise ValueError('No data files found in %s' % (data_source,))
        return data_files

    def parser(self, serialized_example):
        """Parses a single tf.Example into image and label tensors."""
        # Specify how the TF-Examples are decoded.
        keys_to_features = {
            'image/encoded': tf.FixedLenFeature(
                (), tf.string, default_value=''),
            'image/filename': tf.FixedLenFeature(
                (), tf.string, default_value=''),
            'image/format': tf.FixedLenFeature(
                (), tf.string, default_value='jpeg'),
            'image/height': tf.FixedLenFeature(
                (), tf.int64, default_value=0),
            'image/width': tf.FixedLenFeature(
                (), tf.int64, default_value=0),
            'image/segmentation/class/encoded': tf.FixedLenFeature(
                (), tf.string, default_value=''),
            'image/segmentation/class/format': tf.FixedLenFeature(
                (), tf.string, default_value='png'),
        }

        feature = tf.parse_single_example(serialized_example, keys_to_features)

        height = tf.cast(feature['image/height'], tf.int32)
        width = tf.cast(feature['image/width'], tf.int32)

        image = tf.image.decode_image(
            feature['image/encoded'], channels=3)
        image = tf.reshape(image, [height, width, 3])

        label = tf.image.decode_image(
            feature['image/segmentation/class/encoded'], channels=1)

        label = tf.reshape(label, [height, width, 1])

        # preprocess
        image, label = input_preprocess.preprocess_image_and_label(
            image,
            label,
            self.model_input_height,
            self.model_input_width,
            self.min_scale_factor,
            self.max_scale_factor,
            self.scale_factor_step_size,
            _DATASETS_INFORMATION[self.dataset_name].ignore_label,
            self.is_training)
        # image/label : [model_input_height, model_input_width, 3]
        return image, label

    def make_batch(self, batch_size, num_epochs=1):
        filenames = self._get_filenames()
        dataset = tf.data.TFRecordDataset(filenames).repeat(num_epochs)

        dataset = dataset.map(self.parser, num_parallel_calls=batch_size)

        # Potentially shuffle records.
        if self.subset == 'train':
            min_queue_examples = int(
                _DATASETS_INFORMATION[self.dataset_name].
                subset_to_sizes[self.subset] * 0.4)
            # Ensure that the capacity is sufficiently large to provide
            # good random shuffling.
            dataset = dataset.shuffle(
                buffer_size=min_queue_examples + 3 * batch_size)

        # batch
        dataset = dataset.prefetch(batch_size)
        dataset = dataset.batch(batch_size)
        iterator = dataset.make_one_shot_iterator()
        return iterator.get_next()

    def show_image(self):
        filenames = self._get_filenames()
        dataset = tf.data.TFRecordDataset(filenames)

        dataset = dataset.map(self.parser)
        for image, label in dataset:
            image_shape = tf.shape(image)
            image = tf.cast(image, tf.uint8)
            label = tf.squeeze(label)
            print(image_shape)
            fig = plt.figure()
            fig.add_subplot(1, 2, 1)
            plt.imshow(image)
            fig.add_subplot(1, 2, 2)
            plt.imshow(label)
            plt.show()
            break


