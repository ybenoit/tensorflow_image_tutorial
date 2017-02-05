# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Makes helper libraries available in the cifar10 package."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class Constant:
    def __init__(self):
        pass

    DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'

    # If a model is trained with multiple GPUs, prefix all Op names with tower_name
    # to differentiate the operations. Note that this prefix is removed from the
    # names of the summaries when visualizing a model.
    TOWER_NAME = 'tower'

    IMAGE_SIZE = 28

    NUM_CLASSES = 10

    NUM_PIXELS = IMAGE_SIZE * IMAGE_SIZE

    NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000

    NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000
