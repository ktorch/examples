
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

The `res.q <https://github.com/ktorch/examples/blob/master/vision/res.q>`_ scripts loads
`cifar.q <https://github.com/ktorch/examples/blob/master/vision/cifar.q>`_ to read in CIFAR10 data.

::

   q)show d:cifar10[]
   x| (((59 43 50 68 98 119 139 145 149 149 131 125 142 144 137 129 137 134 124 ..
   y| 6 9 9 4 1 1 2 7 8 3 4 7 7 2 9 9 9 3 2 6 4 3 6 6 2 6 3 5 4 0 0 9 1 3 4 0 3 ..
   X| (((158 159 165 166 160 156 162 159 158 159 161 160 161 166 169 170 167 162..
   Y| 3 8 8 0 6 6 1 6 3 1 0 9 5 7 9 8 5 7 8 6 7 0 4 9 5 2 4 0 9 6 6 5 4 5 9 2 4 ..
   s| `airplane`automobile`bird`cat`deer`dog`frog`horse`ship`truck

Dictionary keys ```x`` & ```y`` contain the training data, 50,000 32x32 images and the corresponding class, 0-9.
Upper case keys ```X`` & ```Y`` contain the test data, 10,000 images and classes.

::

   q)count each d`x`y
   50000 50000

   q)count each d`X`Y
   10000 10000

The mean and standard deviation of the training data images,
calculated for each red,green and blue channel,
is used to standardize the data to mean zero and a standard deviation of one.
The training data is flipped horizontally to make 100,000 training images and classes:

::

   d[`mean`std]:meanstd(d`x;0 2 3)      /calculate mean & stddev by RGB channel
   @[`d;`x`X;{zscore(x;d`mean;d`std)}]; /standardize train & test data to mean zero, stddev of 1
   q)d[`x],:Flip(d`x;-1)                /add horizontal flip of each training image
   q)d[`y],:d`y                         /repeat training targets

The `resnet.q <https://github.com/ktorch/examples/blob/master/vision/resnet.q>`_ script define the ResNet blocks.

The basic and bottleneck blocks take input channels, output channels, stride and expansion factor:

::

   q)basic[3;64;1;1]    /3 in, 64 output channels, stride=1, expansion factor=1
   `sequential`basic
   ,(`conv2d;`conv1;3;64;3;1;1;(`bias;0b))
   ,(`batchnorm2d;`bn1;64)
   ,(`relu;`relu1;1b)
   ,(`conv2d;`conv2;64;64;3;1;1;(`bias;0b))
   ,(`batchnorm2d;`bn2;64)

   q)bottle[3;64;4;1]   /3 in, 64 output channels, stride=1, expansion factor=4
   `sequential`bottleneck
   ,(`conv2d;`conv1;3;64;1;1;(`bias;0b))
   ,(`batchnorm2d;`bn1;64)
   ,(`relu;`relu1;1b)
   ,(`conv2d;`conv2;64;64;3;1;1;(`bias;0b))
   ,(`batchnorm2d;`bn2;64)
   ,(`relu;`relu2;1b)
   ,(`conv2d;`conv3;64;256;1;1;(`bias;0b))
   ,(`batchnorm2d;`bn3;256)

These blocks are grouped in 4 layers to create a ResNet module, e.g.

::

   q)show q:resnet[`basic; 1b; 2 2 2 2; 10]  / basic blocks, true for alternate case, 4 layers, 2 deep, 10 classes
   `sequential`resnet
   ,(`conv2d;`conv1;3;64;3;1;1;(`bias;0b))
   ,(`batchnorm2d;`bn1;64)
   ,(`relu;`relu;1b)
   (`seqnest`layer1;(`residual;(`sequential`basic;,(`conv2d;`conv1;64;64;3;1;1;(..
   (`seqnest`layer2;(`residual;(`sequential`basic;,(`conv2d;`conv1;64;128;3;2;1;..
   (`seqnest`layer3;(`residual;(`sequential`basic;,(`conv2d;`conv1;128;256;3;2;1..
   (`seqnest`layer4;(`residual;(`sequential`basic;,(`conv2d;`conv1;256;512;3;2;1..
   ,(`adaptavg2d;`avgpool;1 1)
   ,(`flatten;`flatten;1)
   ,(`linear;`fc;512;10)

   q)q:module q  /define and allocate a PyTorch module in c++ from k definitions

Training the model for 60 epochs (approximately 43 seconds per epoch on a single GTX 1080 gpu) takes around 42 minutes and manages about 92% accuracy on the test images.

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
    
Building a table of results, the main source of misclassification is the model mistaking cats for dogs and vice versa:

::

   q)show 5?t:d[`s]@/:([]y:d`Y; yhat:evaluate(m;V;1000;`max))
   y          yhat      
   ---------------------
   automobile automobile
   airplane   airplane  
   horse      horse     
   truck      automobile
   ship       ship      

   q)select[10;>n] n:count i by y,yhat from t where not y=yhat
   y          yhat      | n 
   ---------------------| --
   cat        dog       | 86
   dog        cat       | 68
   cat        deer      | 31
   truck      automobile| 30
   horse      dog       | 28
   cat        frog      | 28
   cat        bird      | 27
   automobile truck     | 26
   bird       deer      | 25
   bird       airplane  | 24


Wide ResNet model
*****************

The `wide.q <https://github.com/ktorch/examples/blob/master/vision/wide.q>`_ script creates a newer form of the ResNet model,
decreasing depth and increasing width.
After training for ?? epochs, it achieves 96 - 97% accuracy in about xx hours.
In addition to an improved model, the script augments the data by using random cropping in additional to random horizontal
flips of the training images.  The loss model is smoothed cross entropy in an attempt to prevent the model from overfitting to the training data at the expense of generalizing the parameters for classifying out-of-sample images.

Running the wide ResNet for 200 epochs on a single GPU takes about a minute per epoch, running for 3.xx hours trains a model with about 97% accuracy.  Again the main source of confusion is over cats and dogs:


