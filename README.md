# TensorFlow Image Tutorial

This repository contains examples neural network implementations in TensorFlow for image classification.
The code structure aims at showing that it is easy to modify the structure of a neural network without impacting the rest of the code.

The examples run with TensorFlow 1.0, with the Python API.

## Download the data
```
bash script/launch_data_extraction.sh
```

## Run the code
Example for the softmax neural network :
```
bash script/launch_softmax_training.sh
```
Choose the model you want and run the corresponding Python file.

## See results in TensorBoard
```
tensorboard --logdir="/tmp/not_mnist/not_mnist_logs"
```

