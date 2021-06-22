
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
