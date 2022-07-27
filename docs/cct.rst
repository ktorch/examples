.. _cct:

Compact Convolutional Transformer (CCT)
=======================================
Applying transformers and multi-headed self-attention to vision has involved training on large data sets with a lot of GPU's in parallel.
The designers of the `Compact Convolutional Transformer <https://github.com/SHI-Labs/Compact-Transformers>`_ aimed for a design
that could be trained with a single GPU on a small dataset, e.g MNIST or CIFAR10.

The image is made into tokens by one or two convolutional layers, then passed through a position embedding layer. From there, a set of transformer layers,
ending with two linear layers to ouput the logits for classification.

Scripts
*******

- `mnist/cct.q <https://github.com/ktorch/examples/blob/master/mnist/cct.q>`_ - small CCT model for MNIST data with 2 transformer layers, 280K parameters
- `vision/cct.q <https://github.com/ktorch/examples/blob/master/vision/cct.q>`_ - larger CCT model for CIFAR10 data with 6 transformer layers, 3.2 million parameters

CCT model
*********

The CCT model implemented in the example scripts is made up of the following parts:

- input transformations
- tokens from 1-2 convolutions
- position embedding
- encoding transformer blocks with self-attention
- decode block with an attention "pooling" layer and a final linear layer

Input transformations
^^^^^^^^^^^^^^^^^^^^^
The example scripts have an initial module to perform some basic augmentation of the training data.  (see PyTorch's vision documentation on the variety of augmentations used to create more training examples from the small training data of MNIST & CIFAR10).

e.g. for MNIST data, which is a single channel of 28 x 28 images, the training data is cropped after padding with a 3-pixel boundary. Both the training and test data are transformed to having zero mean and unit standard deviation by subtracting the mean of the pixels of the training data, then dividing by the training data's standard deviation:

::

   q)inp
   {
    r:((`randomcrop;`crop;x;y);(`zscore;`zscore;z 0;z 1));
    enlist[`transform`input],{seq`sequential,x}'[2 -1#\:r]}

   q)inp[28;3;d`mean`std]
   `transform`input
   (`sequential;,(`randomcrop;`crop;28;3);,(`zscore;`zscore;,33.32e;,78.57e))
   (`sequential;,(`zscore;`zscore;,33.32e;,78.57e))

   q)-2 str m:module inp[28;3;d`mean`std];
   Transform((
     (train): torch::nn::Sequential(
       (crop): knn::RandomCrop(size=[28, 28], pad=[3, 3, 3, 3])
       (zscore): knn::Zscore(mean=33.3184, stddev=78.5675, inplace=false)
     )
     (eval): torch::nn::Sequential(
       (zscore): knn::Zscore(mean=33.3184, stddev=78.5675, inplace=false)
     )
   )

Tokens
^^^^^^
The CCT model uses 1 or 2 convolutional layers to create tokens of the image:

::

   q)tok
   {
    s:("conv";"maxpool";"relu"); s:`$$[x=1;s;s,\:/:"12"];
    $[x=1; conv[s;y;z]; conv[s 0;y;64],conv[s 1;64;z]],
     ((`flatten;`flat;2;3); `transpose`transpose)}

   q)tok[2;1;28]  /2 convolutional layers for single channel MNIST images
   (`conv2d;`conv1;1;64;3;1;1;(`bias;0b))
   `relu`maxpool1
   (`maxpool2d;`relu1;3;2;1)
   (`conv2d;`conv2;64;28;3;1;1;(`bias;0b))
   `relu`maxpool2
   (`maxpool2d;`relu2;3;2;1)
   (`flatten;`flat;2;3)
   `transpose`transpose

   q)tok[1;3;32] / 1 convolutional layer for 3-channel 32x32 CIFAR10 images
   (`conv2d;`conv;3;32;3;1;1;(`bias;0b))
   `relu`maxpool
   (`maxpool2d;`relu;3;2;1)
   (`flatten;`flat;2;3)
   `transpose`transpose

The convolutional layers, together with a max pooling layer and a ReLU activiation are bundled in a sequential container:

::

   q)q:module`seqnest,enlist'[tok[2;1;128]]

   q)-2 str q;
   knn::SeqNest(
     (conv1): torch::nn::Conv2d(1, 64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], bias=false)
     (maxpool1): torch::nn::ReLU()
     (relu1): torch::nn::MaxPool2d(kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1], ceil_mode=false)
     (conv2): torch::nn::Conv2d(64, 128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], bias=false)
     (maxpool2): torch::nn::ReLU()
     (relu2): torch::nn::MaxPool2d(kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1], ceil_mode=false)
     (flat): torch::nn::Flatten(start_dim=2, end_dim=3)
     (transpose): knn::Transpose(dim0=-2, dim1=-1)
   )


Position embedding
^^^^^^^^^^^^^^^^^^

A learnable position embedding layer is added to the previous convolutional tokens.
To build the embedding requires the embedding dimension (columns), the number of convolutions and the image size.
For example, for 128 embedding dimension, 2 convolutions and 28x28 images:

::

   q)pos:{[e;t;s](`residual`position; (`sequential; enlist(`embedpos;`emb;n*n:s div 2*t;e)))}

   q)pos[128;2;28]
   residual    position                
   `sequential ,(`embedpos;`emb;49;128)

   q)-2 str p:module pos[128;2;28]
   knn::Residual(
     (q1): torch::nn::Sequential(
       (emb): knn::EmbedPosition(rows=49, cols=128)
     )
   )

The number of rows for the embedding (the sequence length) is calculated from the number of convolutions and the image size:

For example, using the single channel 28x28 pixel images of MNIST and 2 convolutional tokenizing layers, 
there are 49 positions:

::

   q)x:tensor(`randn;10 1 28 28)
   q)q:module`seqnest,enlist'[tok[2;1;128]]
   q)y:forward(q;x)
   q)size y
   10 49 128

   q){x*x}28 div 2*2  / square of image size divided by 2 times the number of convolutional layers
   49

For 3-channel 32x32 pixel CIFAR-10 images and one convolutional layer for tokenizing:

::

   q)x:tensor(`randn;10 3 32 32)
   q)q:module`seqnest,enlist'[tok[1;3;128]]
   q)size y:forward(q;x)
   10 256 128

   q){x*x}32 div 2*1 / square of image size divided by twice the number of tokenizing layers
   256

Transformer block
^^^^^^^^^^^^^^^^^

::

   e:256            /embedding dimension
   h:4              /heads for self-attention
   m:2              /multiplier of embed dim for linear layer in encoder
   p1:.1; p2:.05    /dropout probabilities

   q)blk
   {[e;h;m;p1;p2]
    a:seq(`sequential; (`selfattention;`attn;e;h;p1;1b); (`droppath;`drop;p2));
    b:seq(`sequential; (`layernorm;`norm;e); (`linear;`linear1;e;e*m); `gelu`gelu; (`linear;`linear2;e*m;e); (`droppath;`drop;p2));
    (`sequential; (`residual`resid1;a); (`residual`resid2;b))}

The encoder block is a sequential with two residual layers where the result is ``x+f(x)``; the 1st residual layer is the self-attention layer, followed by a 2nd residual layer with linear layers and a GELU activation function in between.

::

   q)blk[e;h;m;p1;p2]
   `sequential
   (`residual`resid1;(`sequential;,(`selfattention;`attn;256;4;0.1;1b);,(`droppa..
   (`residual`resid2;(`sequential;,(`layernorm;`norm;256);,(`linear;`linear1;256..

   q)-2 str q:module blk[e;h;m;p1;p2]
   torch::nn::Sequential(
     (resid1): knn::Residual(
       (q1): torch::nn::Sequential(
         (attn): knn::SelfAttention(dim=256, heads=4, dropout=0.1, norm=true)(
           (norm): torch::nn::LayerNorm([256], eps=1e-05, elementwise_affine=true)
           (in): torch::nn::Linear(in_features=256, out_features=768, bias=false)
           (drop): torch::nn::Dropout(p=0.1, inplace=false)
           (out): torch::nn::Linear(in_features=256, out_features=256, bias=true)
         )
         (drop): knn::DropPath(p=0.05)
       )
     )
     (resid2): knn::Residual(
       (q1): torch::nn::Sequential(
         (norm): torch::nn::LayerNorm([256], eps=1e-05, elementwise_affine=true)
         (linear1): torch::nn::Linear(in_features=256, out_features=512, bias=true)
         (gelu): torch::nn::GELU()
         (linear2): torch::nn::Linear(in_features=512, out_features=256, bias=true)
         (drop): knn::DropPath(p=0.05)
       )
     )
   )

The encoder block uses both a `dropout <https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html>`_ layer that is implemented in PyTorch,
together with `droppath <https://ktorch.readthedocs.io/en/latest/kmodules.html#module-droppath>`_ layers that are implemented as part of the k api since this type of dropout is not part of PyTorch in either the python or C++ libraries.

The ``droppath`` layers are given an increasing probability of dropping or setting to zero some of the batch inputs.
The ``enc`` function calls ``blk`` for each layer and handles the increasing probability of the ``droppath`` layers:

::

  enc:{[e;h;m;n;p]enlist[`seqlist`blocks],blk[e;h;m;p]'[(n-1) ((p%n-1)+)\0.0]}


Decoder block
^^^^^^^^^^^^^

The final block of the CCT module "pools" the attention layer output:

- The ``b x n x d`` output is passed through a ``d x 1`` linear layer.
- The result is a ``b x n x 1`` array (b-batches, n-sequence length or number of tokens, d - embed dimension).
- From there, the ``b x n x 1`` result is run through a softmax layer across the sequence dimension to give a weight to each of the tokens.
- The result is transposed to ``b x 1 x n`` and multiplied by the original output of the attention layer (``b x n x d``) to produce a ``b x 1 x d`` tensor.
- The 2nd dimension is squeezed out, with the ``b x d`` result multiplied by a final linear ``d x k`` layer to return a matrix with one row ber batch and one column for each of the possible ``k`` classes.

The ``dec`` function uses the embedding dimension and the number of classes to create the final decoding block of the CCT model:

::

   q)dec
   {[e;k]
    (`seqnest`end;
      enlist(`layernorm;`norm;e);
     (`seqjoin`join; seq(`sequential; (`linear;`attnpool;e;1); (`softmax;`softma..
      enlist(`squeeze;`squeeze;-2);
      enlist(`linear;`fc;e;k))}

   q)dec[128;10]
   `seqnest`end
   ,(`layernorm;`norm;128)
   (`seqjoin`join;(`sequential;,(`linear;`attnpool;128;1);,(`softmax;`softmax;1)..
   ,(`squeeze;`squeeze;-2)
   ,(`linear;`fc;128;10)

   q)-2 str q:module dec[128;10];
   knn::SeqNest(
     (norm): torch::nn::LayerNorm([128], eps=1e-05, elementwise_affine=true)
     (join): knn::SeqJoin(
       (qx): torch::nn::Sequential(
         (attnpool): torch::nn::Linear(in_features=128, out_features=1, bias=true)
         (softmax): torch::nn::Softmax(dim=1)
         (transpose): knn::Transpose(dim0=-2, dim1=-1)
       )
       (mul): knn::Matmul()
     )
     (squeeze): knn::Squeeze(dim=-2, inplace=false)
     (fc): torch::nn::Linear(in_features=128, out_features=10, bias=true)
   )


Full model
^^^^^^^^^^

The full CCT model is initially built as a dictionary of trees for each of the parts:

::

   q:`inp`tok`pos`enc`dec!()
   q.inp: inp[s;4;d`mean`std]
   q.tok: seq enlist[`seqnest`token],tok[t;i;e]        /tokens from convolution(s)
   q.pos: pos[e;t;s]                                   /position embedding
   q.enc: enc[e;h;m;n;p]                               /encoder blocks
   q.dec: dec[e;k]                                     /decoder layers

The parts are added to a parent sequential model at depth 1:

::

   q:{module(x;1;module y); x}/[module`sequential;q]

(the intermediate module created is automatically free'd by the module call with parent)


MNIST script
************

The `mnist/cct.q <https://github.com/ktorch/examples/blob/master/mnist/cct.q>`_ script builds a CCT model for MNIST data with 2 transformer layers and 280,655 parameters. Training on a single NVIDIA GeForce GTX 1080 Ti GPU runs abount 6 seconds per epoch using a batch size of 50. After 40 epochs (around 4 minutes) test accuracy is 99.5% - 99.6% with minimal augmentation.

See a PyTorch representation of the full `model  <https://github.com/ktorch/examples/blob/master/mnist/out/cct.txt>`_ along with a sample training
`log <https://github.com/ktorch/examples/blob/master/mnist/out/cct.log>`_.


CIFAR-10 script
***************

The `vision/cct.q <https://github.com/ktorch/examples/blob/master/vision/cct.q>`_ has parameters for both the small, 2-layer model used with MNIST digits and a larger 6-layer model. Both can be trained with the CIFAR-10 dataset with test accuracy from 88% - 92%. More data augmentation is needed to bring the accuracy near the 95-98% achieved in the paper. See the :ref:`vision` section for more on the CIFAR-10 dataset.
