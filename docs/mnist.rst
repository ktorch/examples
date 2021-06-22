.. _mnist:

MNIST
=====

Scripts
*******

- `mnist.q <https://github.com/ktorch/examples/blob/master/mnist/mnist.q>`_ - script to read in MNIST data
- `conv.q <https://github.com/ktorch/examples/blob/master/mnist/conv.q>`_ - convolutional model to classify MNIST digits
- `gan.q <https://github.com/ktorch/examples/blob/master/mnist/gan.q>`_ - generative adversarial network to generate digits
- `lstm.q <https://github.com/ktorch/examples/blob/master/mnist/lstm.q>`_ - alternate, recurrent form of model to classify digits

Downloading the data
********************

The MNIST dataset is available `here <http://yann.lecun.com/exdb/mnist/>`_.
The `mnist.q <https://github.com/ktorch/examples/blob/master/mnist/mnist.q>`_ - script assumes the downloaded binary files are uncompressed in a ``data/`` directory that exists at the same level as the script with files:

::

   examples/mnist
   examples/mnist/mnist.q
   examples/mnist/conv.q
   examples/mnist/gan.q
   examples/mnist/lstm.q
   examples/mnist/data
   examples/mnist/data/t10k-images-idx3-ubyte
   examples/mnist/data/t10k-labels-idx1-ubyte
   examples/mnist/data/train-images-idx3-ubyte
   examples/mnist/data/train-labels-idx1-ubyte

Loading the data into k/q
*************************
