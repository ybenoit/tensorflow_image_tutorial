from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf

from tensorflow_image_tutorial.input import DataExtractor

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string("data_dir",
                           os.path.join(os.path.dirname(__file__), "../data/NotMNIST/"),
                           """Directory containing the input data""")


def main():
    """
    Download and extract NoMNIST Data
    """

    # Download and extract data
    data_extractor = DataExtractor(
        data_url='http://yaroslavvb.com/upload/notMNIST/',
        num_classes=10,
        image_size=28,
        pixel_depth=255.0,
        train_size=200000,
        valid_size=10000,
        test_size=10000,
        data_dir=FLAGS.data_dir,
        pickle_file='notMNIST.pickle')

    data_extractor.extract()


if __name__ == '__main__':
    main()
