
.. _vision:

Vision
======

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

   > find examples/vision/data
   examples/vision/data
   examples/vision/data/readme.md
   examples/vision/data/cifar-10-batches-bin
   examples/vision/data/cifar-10-batches-bin/batches.meta.txt
   examples/vision/data/cifar-10-batches-bin/data_batch_1.bin
   examples/vision/data/cifar-10-batches-bin/data_batch_2.bin
   examples/vision/data/cifar-10-batches-bin/data_batch_3.bin
   examples/vision/data/cifar-10-batches-bin/data_batch_4.bin
   examples/vision/data/cifar-10-batches-bin/data_batch_5.bin
   examples/vision/data/cifar-10-batches-bin/test_batch.bin
   examples/vision/data/cifar-10-batches-bin/readme.html


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
calculated for each red, green and blue channel,
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

     1.  lr: 0.080000  loss: 1.395746  test: 1.0029  accuracy: 64.97%   19:00:52
     2.  lr: 0.079945  loss: 0.764869  test: 0.7085  accuracy: 74.70%   19:01:34
     3.  lr: 0.079781  loss: 0.538976  test: 0.5815  accuracy: 79.82%   19:02:16
     4.  lr: 0.079508  loss: 0.424581  test: 0.4919  accuracy: 83.47%   19:02:58
     5.  lr: 0.079126  loss: 0.364722  test: 0.5597  accuracy: 81.40%   19:03:41
     6.  lr: 0.078637  loss: 0.324985  test: 0.4836  accuracy: 83.63%   19:04:23
     7.  lr: 0.078042  loss: 0.292052  test: 0.4947  accuracy: 83.65%   19:05:06
   ..
    58.  lr: 0.000492  loss: 0.001877  test: 0.2648  accuracy: 92.00%   19:41:14
    59.  lr: 0.000219  loss: 0.001876  test: 0.2646  accuracy: 92.08%   19:41:56
    60.  lr: 0.000055  loss: 0.001868  test: 0.2654  accuracy: 92.04%   19:42:39


Building a table of results, the main source of misclassification is the model mistaking cats for dogs and vice versa:

::

   q)test(m;`metrics;`predict)

   q)show 5?t:d[`s]@/:([]y:d`Y; yhat:testrun m)

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
   dog        cat       | 82
   cat        dog       | 65
   cat        bird      | 30
   bird       airplane  | 30
   truck      automobile| 27
   airplane   ship      | 26
   bird       deer      | 25
   frog       cat       | 24
   automobile truck     | 22
   cat        frog      | 22


Wide ResNet model
*****************

The `wide.q <https://github.com/ktorch/examples/blob/master/vision/wide.q>`_ script creates a newer form of the ResNet model,
decreasing depth and increasing width.
In addition to an improved model, the script augments the data by using random cropping in additional to random horizontal
flips of the training images.  The loss model is smoothed cross entropy in an attempt to prevent the model from overfitting to the training data at the expense of generalizing the parameters for classifying out-of-sample images.

Running the wide ResNet for 200 epochs on a single GPU takes about a minute per epoch, running for about 3.5 hours to train a model with about 96% accuracy.  

::

   > q examples/vision/wide.q
   KDB+ 4.0 2020.05.04 Copyright (C) 1993-2020 Kx Systems
   l64/ 12(16)core 64037MB 

     1.  lr: 0.080000  loss: 1.635264  test: 1.5421  accuracy: 53.36%   15:32:08
     2.  lr: 0.079995  loss: 1.240133  test: 1.2023  accuracy: 69.17%   15:33:09
     3.  lr: 0.079980  loss: 1.083723  test: 1.1974  accuracy: 70.48%   15:34:10
     4.  lr: 0.079956  loss: 0.994642  test: 1.0060  accuracy: 78.84%   15:35:11
     5.  lr: 0.079921  loss: 0.946601  test: 0.9948  accuracy: 78.95%   15:36:13
     6.  lr: 0.079877  loss: 0.908107  test: 0.9804  accuracy: 79.39%   15:37:14
     7.  lr: 0.079822  loss: 0.881616  test: 0.9139  accuracy: 83.34%   15:38:16
   ..
    98.  lr: 0.041884  loss: 0.631557  test: 0.7380  accuracy: 91.01%   17:12:12
    99.  lr: 0.041256  loss: 0.630563  test: 0.7389  accuracy: 90.55%   17:13:14
   100.  lr: 0.040628  loss: 0.627417  test: 0.7118  accuracy: 92.03%   17:14:15
   101.  lr: 0.040000  loss: 0.629066  test: 0.7673  accuracy: 90.15%   17:15:16
   ..
   195.  lr: 0.000178  loss: 0.502801  test: 0.6094  accuracy: 96.13%   18:51:12
   196.  lr: 0.000123  loss: 0.502723  test: 0.6111  accuracy: 96.17%   18:52:13
   197.  lr: 0.000079  loss: 0.502722  test: 0.6108  accuracy: 96.10%   18:53:15
   198.  lr: 0.000044  loss: 0.502749  test: 0.6112  accuracy: 96.15%   18:54:16
   199.  lr: 0.000020  loss: 0.502889  test: 0.6126  accuracy: 96.21%   18:55:17
   200.  lr: 0.000005  loss: 0.502738  test: 0.6099  accuracy: 96.17%   18:56:18


Again the main source of confusion is cats and dogs:

::


   q)test(m;`metrics;`predict)
   q)t:d[`s]@/:([]y:d`Y; yhat:testrun m)

   q)select[10;>n] n:count i by y,yhat from t where not y=yhat
   y          yhat      | n 
   ---------------------| --
   cat        dog       | 50
   dog        cat       | 36
   automobile truck     | 19
   airplane   ship      | 19
   bird       frog      | 15
   ship       airplane  | 14
   bird       deer      | 12
   truck      automobile| 11
   cat        bird      | 11
   dog        bird      | 11

