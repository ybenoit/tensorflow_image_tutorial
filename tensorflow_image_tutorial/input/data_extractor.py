from __future__ import print_function

import numpy as np
import os
import sys
import tarfile
from scipy import ndimage
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle


class DataExtractor:
    def __init__(self, data_url, num_classes, image_size, pixel_depth, train_size, valid_size, test_size, data_dir,
                 pickle_file):
        self.data_url = data_url
        self.num_classes = num_classes
        self.image_size = image_size
        self.pixel_depth = pixel_depth
        self.train_size = train_size
        self.valid_size = valid_size
        self.test_size = test_size
        self.data_dir = data_dir
        self.pickle_file = pickle_file

    def extract(self):
        # Download Data
        train_filename = self.maybe_download(self.data_url, self.data_dir, 'notMNIST_large.tar.gz', 247336696)
        test_filename = self.maybe_download(self.data_url, self.data_dir, 'notMNIST_small.tar.gz', 8458043)

        # Extract Data
        train_folders = self.maybe_extract(self.data_dir, train_filename, self.num_classes)
        test_folders = self.maybe_extract(self.data_dir, test_filename, self.num_classes)

        # Pickle
        train_datasets = self.maybe_pickle(train_folders, 45000)
        test_datasets = self.maybe_pickle(test_folders, 1800)

        # Merge datasets
        valid_dataset, valid_labels, train_dataset, train_labels = self.merge_datasets(
            train_datasets, self.image_size, self.train_size, self.valid_size)
        _, _, test_dataset, test_labels = self.merge_datasets(test_datasets, self.image_size, self.test_size)

        print('Training:', train_dataset.shape, train_labels.shape)
        print('Validation:', valid_dataset.shape, valid_labels.shape)
        print('Testing:', test_dataset.shape, test_labels.shape)

        # Randomize datasets
        train_dataset, train_labels = self.randomize(train_dataset, train_labels)
        test_dataset, test_labels = self.randomize(test_dataset, test_labels)
        valid_dataset, valid_labels = self.randomize(valid_dataset, valid_labels)

        # Save data
        try:
            f = open(os.path.join(self.data_dir, self.pickle_file), 'wb')
            save = {
                'train_dataset': train_dataset,
                'train_labels': train_labels,
                'valid_dataset': valid_dataset,
                'valid_labels': valid_labels,
                'test_dataset': test_dataset,
                'test_labels': test_labels,
            }
            pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
            f.close()
        except Exception as e:
            print('Unable to save data to', os.path.join(self.data_dir, self.pickle_file), ':', e)
            raise

        statinfo = os.stat(os.path.join(self.data_dir, self.pickle_file))
        print('Compressed pickle size:', statinfo.st_size)

    @staticmethod
    def maybe_download(url, data_dir, filename, expected_bytes, force=False):
        """
        Download a file if not present, and make sure it's the right size.
        """
        if force or not os.path.exists(os.path.join(data_dir, filename)):
            filename, _ = urlretrieve(url + filename, os.path.join(data_dir, filename))
        statinfo = os.stat(os.path.join(data_dir, filename))
        if statinfo.st_size == expected_bytes:
            print('Found and verified', filename)
        else:
            raise Exception(
                'Failed to verify' + filename + '. Can you get to it with a browser?')
        return filename

    @staticmethod
    def maybe_extract(data_dir, filename, num_classes, force=False):
        root = os.path.splitext(os.path.splitext(filename)[0])[0]  # remove .tar.gz
        if os.path.isdir(os.path.join(data_dir, root)) and not force:
            # You may override by setting force=True.
            print('%s already present - Skipping extraction of %s.' % (root, filename))
        else:
            print('Extracting data for %s. This may take a while. Please wait.' % root)
            tar = tarfile.open(os.path.join(data_dir, filename))
            sys.stdout.flush()
            tar.extractall(path=data_dir)
            tar.close()
        data_folders = [
            os.path.join(data_dir, root, d) for d in sorted(os.listdir(os.path.join(data_dir, root)))
            if os.path.isdir(os.path.join(data_dir, root, d))]
        if len(data_folders) != num_classes:
            raise Exception(
                'Expected %d folders, one per class. Found %d instead.' % (
                    num_classes, len(data_folders)))
        print(data_folders)
        return data_folders

    @staticmethod
    def load_letter(folder, min_num_images, image_size, pixel_depth):
        """Load the data for a single letter label."""
        image_files = os.listdir(folder)
        dataset = np.ndarray(shape=(len(image_files), image_size, image_size),
                             dtype=np.float32)
        image_index = 0
        print(folder)
        for image in os.listdir(folder):
            image_file = os.path.join(folder, image)
            try:
                image_data = (ndimage.imread(image_file).astype(float) -
                              pixel_depth / 2) / pixel_depth
                if image_data.shape != (image_size, image_size):
                    raise Exception('Unexpected image shape: %s' % str(image_data.shape))
                dataset[image_index, :, :] = image_data
                image_index += 1
            except IOError as e:
                print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')

        num_images = image_index
        dataset = dataset[0:num_images, :, :]
        if num_images < min_num_images:
            raise Exception('Many fewer images than expected: %d < %d' %
                            (num_images, min_num_images))

        print('Full dataset tensor:', dataset.shape)
        print('Mean:', np.mean(dataset))
        print('Standard deviation:', np.std(dataset))
        return dataset

    def maybe_pickle(self, data_folders, min_num_images_per_class, force=False):
        dataset_names = []
        for folder in data_folders:
            set_filename = folder + '.pickle'
            dataset_names.append(set_filename)
            if os.path.exists(set_filename) and not force:
                # You may override by setting force=True.
                print('%s already present - Skipping pickling.' % set_filename)
            else:
                print('Pickling %s.' % set_filename)
                dataset = self.load_letter(folder, min_num_images_per_class, self.image_size, self.pixel_depth)
                try:
                    with open(set_filename, 'wb') as f:
                        pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
                except Exception as e:
                    print('Unable to save data to', set_filename, ':', e)

        return dataset_names

    @staticmethod
    def make_arrays(nb_rows, img_size):
        if nb_rows:
            dataset = np.ndarray((nb_rows, img_size, img_size), dtype=np.float32)
            labels = np.ndarray(nb_rows, dtype=np.int32)
        else:
            dataset, labels = None, None
        return dataset, labels

    def merge_datasets(self, pickle_files, image_size, train_size, valid_size=0):
        num_classes = len(pickle_files)
        valid_dataset, valid_labels = self.make_arrays(valid_size, image_size)
        train_dataset, train_labels = self.make_arrays(train_size, image_size)
        vsize_per_class = valid_size // num_classes
        tsize_per_class = train_size // num_classes

        start_v, start_t = 0, 0
        end_v, end_t = vsize_per_class, tsize_per_class
        end_l = vsize_per_class + tsize_per_class
        for label, pickle_file in enumerate(pickle_files):
            try:
                with open(pickle_file, 'rb') as f:
                    letter_set = pickle.load(f)
                    # let's shuffle the letters to have random validation and training set
                    np.random.shuffle(letter_set)
                    if valid_dataset is not None:
                        valid_letter = letter_set[:vsize_per_class, :, :]
                        valid_dataset[start_v:end_v, :, :] = valid_letter
                        valid_labels[start_v:end_v] = label
                        start_v += vsize_per_class
                        end_v += vsize_per_class

                    train_letter = letter_set[vsize_per_class:end_l, :, :]
                    train_dataset[start_t:end_t, :, :] = train_letter
                    train_labels[start_t:end_t] = label
                    start_t += tsize_per_class
                    end_t += tsize_per_class
            except Exception as e:
                print('Unable to process data from', pickle_file, ':', e)
                raise

        return valid_dataset, valid_labels, train_dataset, train_labels

    @staticmethod
    def randomize(dataset, labels):
        permutation = np.random.permutation(labels.shape[0])
        shuffled_dataset = dataset[permutation, :, :]
        shuffled_labels = labels[permutation]
        return shuffled_dataset, shuffled_labels
