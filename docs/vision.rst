
.. _vision:

Vision models
=============

`Residual networks <https://arxiv.org/abs/1512.03385>`_ and
`wide residual models <https://arxiv.org/abs/1605.07146>`_ are implemented and tested on 
`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ image data.

Scripts
*******

- `cifar.q <https://github.com/ktorch/examples/blob/master/vision/cifar.q>`_ - script to read in CIFAR10 data
- `resnet.q <https://github.com/ktorch/examples/blob/master/vision/resnet.q>`_ - defines the ResNet basic & bottleneck blocks and overall layers
- `widenet.q <https://github.com/ktorch/examples/blob/master/vision/widenet.q>`_ - defines wide resnet with some later improvements
- `res.q <https://github.com/ktorch/examples/blob/master/vision/res.q>`_ - builds and trains a small ResNet model on CIFAR10 data
- `wide.q <https://github.com/ktorch/examples/blob/master/vision/wide.q>`_ - builds and trains a wide ResNet model

Downloading the data
********************
The scripts use `the binary version of the CIFAR10 dataset <https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz>`_,
available as a gzipped tar file. The loader script assumes a data directory exists in the same location as the scripts,
with the following structure for the uncompressed files:

::

   > find vision/
   vision/
   vision/cifar.q
   vision/res.q
   vision/resnet.q
   vision/widenet.q
   vision/wide.q
   vision/data
   vision/data/cifar-10-batches-bin
   vision/data/cifar-10-batches-bin/data_batch_1.bin
   vision/data/cifar-10-batches-bin/batches.meta.txt
   vision/data/cifar-10-batches-bin/data_batch_2.bin
   vision/data/cifar-10-batches-bin/test_batch.bin
   vision/data/cifar-10-batches-bin/data_batch_3.bin
   vision/data/cifar-10-batches-bin/data_batch_4.bin
   vision/data/cifar-10-batches-bin/readme.html
   vision/data/cifar-10-batches-bin/data_batch_5.bin


Basic ResNet model
******************

::

   > q examples/vision/res.q
   KDB+ 4.0 2020.05.04 Copyright (C) 1993-2020 Kx Systems
   l64/ 12(16)core 64037MB 

     1.  lr: 0.080000  loss: 1.369706  test: 1.0592  accuracy: 63.09%   11:06:38
     2.  lr: 0.079945  loss: 0.736021  test: 0.6761  accuracy: 76.53%   11:07:19
     3.  lr: 0.079781  loss: 0.513555  test: 0.5478  accuracy: 81.17%   11:08:01
     4.  lr: 0.079508  loss: 0.409195  test: 0.5639  accuracy: 81.15%   11:08:43
     5.  lr: 0.079126  loss: 0.349066  test: 0.4890  accuracy: 83.34%   11:09:25
     6.  lr: 0.078637  loss: 0.311041  test: 0.5199  accuracy: 82.74%   11:10:07
     7.  lr: 0.078042  loss: 0.281705  test: 0.5238  accuracy: 83.01%   11:10:49
     8.  lr: 0.077343  loss: 0.261755  test: 0.4784  accuracy: 83.97%   11:11:32
     9.  lr: 0.076542  loss: 0.239726  test: 0.4871  accuracy: 83.98%   11:12:14
    10.  lr: 0.075640  loss: 0.222705  test: 0.4388  accuracy: 85.21%   11:12:57
    ..
    58.  lr: 0.000492  loss: 0.001860  test: 0.2526  accuracy: 92.62%   11:46:53
    59.  lr: 0.000219  loss: 0.001867  test: 0.2529  accuracy: 92.68%   11:47:35
    60.  lr: 0.000055  loss: 0.001862  test: 0.2511  accuracy: 92.66%   11:48:17
   2541882 4195776
    

Wide ResNet model
*****************
