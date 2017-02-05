import os
import numpy as np
from six.moves import cPickle as pickle


class DataLoader:
    def __init__(self, data_dir, image_size, num_labels):
        self.data_dir = data_dir
        self.image_size = image_size
        self.num_labels = num_labels

    def load(self):

        with open(os.path.join(self.data_dir, "notMNIST.pickle"), 'rb') as f:
            save = pickle.load(f)
            train_dataset = save['train_dataset']
            train_labels = save['train_labels']
            valid_dataset = save['valid_dataset']
            valid_labels = save['valid_labels']
            test_dataset = save['test_dataset']
            test_labels = save['test_labels']
            del save  # hint to help gc free up memory

        train_dataset, train_labels = self.reformat(train_dataset, train_labels, self.image_size, self.num_labels)
        valid_dataset, valid_labels = self.reformat(valid_dataset, valid_labels, self.image_size, self.num_labels)
        test_dataset, test_labels = self.reformat(test_dataset, test_labels, self.image_size, self.num_labels)

        return train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels

    @staticmethod
    def reformat(dataset, labels, image_size, num_labels):
        dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
        # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
        labels = (np.arange(num_labels) == labels[:, None]).astype(np.float32)
        return dataset, labels

