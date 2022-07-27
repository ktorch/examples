.. _gpt:

GPT
===
Andrej Karpathy, in an effort to simplify the main parts of GPT (Generative Pre-trained Transformer) models,
wrote `minGPT <https://github.com/karpathy/minGPT>`_ in PyTorch. The model is reimplemented using the k-api along with
some modules created for the k interface:
`residual <https://ktorch.readthedocs.io/en/latest/kmodules.html#module-residual>`_,
`selfattention <https://ktorch.readthedocs.io/en/latest/kmodules.html#module-selfattention>`_,
`seqlist <https://ktorch.readthedocs.io/en/latest/kmodules.html#module-seqlist>`_,
`transform <https://ktorch.readthedocs.io/en/latest/kmodules.html#module-transform>`_ and
`callback <https://ktorch.readthedocs.io/en/latest/kmodules.html#module-callback>`_.

Scripts
*******

- `math.q <https://github.com/ktorch/examples/blob/master/gpt/math.q>`_ - addition by learning integer patterns.
- `char.q <https://github.com/ktorch/examples/blob/master/gpt/char.q>`_ - a character-level language model.
- `callback.q <https://github.com/ktorch/examples/blob/master/gpt/callback.q>`_ - character-level language model using a k callback.

GPT model
*********

The GPT model implemented in the examples is made up of the following parts:

- token embedding
- learned position embedding
- overall sequence embedding
- self attention
- transformer blocks
- decode block
- transforms

Token embedding
^^^^^^^^^^^^^^^

The examples define four dimensions that are relevant for embedding:

- ``w`` batch size
- ``v`` vocabulary size, i.e. the total number of tokens
- ``d`` embedding dimension
- ``n`` maximum length of sequence

The input tensor is a ``w`` x ``n`` tensor of long integers which are indices into a list of tokens.
The learned embedding is a weight matrix of ``v`` rows and ``d`` columns.
The result is a ``w`` x ``n`` x ``d`` tensor.

In `math.q <https://github.com/ktorch/examples/blob/master/gpt/math.q>`_,
the simplest example, integers are both the tokens and the indices:

::

   q)x:(6 7 9 8 1 6 5; 5 7 7 8 1 3 5; 2 0 7 7 0 9 7)

   q)x
   6 7 9 8 1 6 5   / 67 + 98 = 165
   5 7 7 8 1 3 5   / 57 + 78 = 135
   2 0 7 7 0 9 7   / 20 + 77 = 097

   q)([]input:-1_'x; `$"->"; target:1_'x)
   input       x  target     
   --------------------------
   6 7 9 8 1 6 -> 7 9 8 1 6 5
   5 7 7 8 1 3 -> 7 7 8 1 3 5
   2 0 7 7 0 9 -> 0 7 7 0 9 7

For the above sample input, ``w`` = 3, ``n`` = 6, i.e. 3 batches of length 6 (the final digit of the sum is part of the target).
The output of the forward calculation is a ``w`` x ``n`` x ``d`` tensor:

::

   q)v:10; w:3; d:20
   q)e:module enlist(`embed; v; d)

   q)y:forward(e; -1_'x)
   q)size y
   3 6 20


Position embedding
^^^^^^^^^^^^^^^^^^

GPT treats the input as a set, but the order of the sequence is also used via a `learned position embedding <https://ktorch.readthedocs.io/en/latest/kmodules.html#module-embedpos>`_.

The relevant dimensions of the position embedding:

- ``d`` embedding dimension
- ``n`` maximum length of sequence

With ``d`` = 20 and a maximum sequence ``n`` = 6:

::

   q)x:(6 7 9 8 1 6 5; 5 7 7 8 1 3 5; 2 0 7 7 0 9 7)

   q)p:module enlist(`embedpos; n; d)

   q)y:forward(p; -1_'x)
   q)size y
   1 6 20

.. note::

   The position embedding is the same for all batches, i.e. it is only a function of length of the sequence and the embedding dimension.

Sequence embedding
^^^^^^^^^^^^^^^^^^

The `embedseq <https://ktorch.readthedocs.io/en/latest/kmodules.html#module-embedseq>`_ module adds the result of the token embedding to the learned position embedding. It uses the two relevant dimensions of the token embedding, ``v`` - the number of tokens and ``d`` - the dimension or number of atttributes of the embedding, along with ``n`` - the longest sequence used in the model.

Below, the two different embeddings are computed with separate modules, then added:

::

   q)x:-1_'(6 7 9 8 1 6 5; 5 7 7 8 1 3 5; 2 0 7 7 0 9 7)
   q)e:module enlist(`embed; v; d)
   q)p:module enlist(`embedpos; n; d)

   q)size y1:forward(e;x)
   3 6 20
   q)size y2:forward(p;x)
   1 6 20

   q)size y:add(y1;y2)
   3 6 20

The same computation is performed by a single ``embedseq`` module:

::

   q)q:module enlist(`embedseq; v; d; n)

   q)parmnames q
   `tok.weight`pos.pos

   q)w:parm(e;`weight); parm(q;`tok.weight;w)  /use same token embeddings
   q)use[w]parm(p;`pos); parm(q;`pos.pos;w)    /use same position embeddings

   q)yseq:forward(q;x)
   q)tensor[y]~tensor yseq
   1b

For the GPT models used in the example scripts, the embedding is built by the function:

::

   q)emb:{[v;d;n;p]seq(`sequential; (`embedseq;v;d;n); (`drop;p))}

   q)emb[10;128;6;.1]
   `sequential
   ,(`embedseq;10;128;6)
   ,(`drop;0.1)

   q)-2 str m:module emb[10;128;6;.1];
   torch::nn::Sequential(
     (0): knn::EmbedSequence(rows=10, cols=128, length=6)(
       (tok): torch::nn::Embedding(num_embeddings=10, embedding_dim=128)
       (pos): knn::EmbedPosition(rows=6, cols=128)
     )
     (1): torch::nn::Dropout(p=0.1, inplace=false)
   )

The embedding is followed by a dropout layer with a 10% probability of setting any particular input to zero.

Masked self-attention
^^^^^^^^^^^^^^^^^^^^^

The main blocks of the GPT models used in the example scripts have two 
`residual <https://ktorch.readthedocs.io/en/latest/kmodules.html#module-residual>`_ layers: the first one uses a masked self-attention layer.

The self-attention layer first passes input through a normalization layer, then sets query, key and value projections from the normalized input,
sets weights according to the softmax of the dot product of queries and keys. The output is passed through a dropout layer and a final linear 
output projection.

::

   q)v:10; w:3; d:20; p:.1; h:4  / h-heads in multi-head attention

   q)q1:seq(`sequential; (`selfattention;d;h;p;1b); (`drop;p));
   q)q1
   `sequential
   ,(`selfattention;20;4;0.1;1b)
   ,(`drop;0.1)

   q)-2 str q1:module q1;
   torch::nn::Sequential(
     (0): knn::SelfAttention(dim=20, heads=4, dropout=0.1, norm=true)(
       (norm): torch::nn::LayerNorm([20], eps=1e-05, elementwise_affine=true)
       (in): torch::nn::Linear(in_features=20, out_features=60, bias=false)
       (drop): torch::nn::Dropout(p=0.1, inplace=false)
       (out): torch::nn::Linear(in_features=20, out_features=20, bias=true)
     )
     (1): torch::nn::Dropout(p=0.1, inplace=false)
   )

The attention is masked so that only tokens up until the most current in the sequence are used.
The upper triangular matrix is created via `triu <https://pytorch.org/docs/stable/generated/torch.triu.html>`_:

::

   q)n:6  / max sequence is 6 tokens for math.q

   q)triu((n,n)#-0w; 1)
   0 -0w -0w -0w -0w -0w
   0 0   -0w -0w -0w -0w
   0 0   0   -0w -0w -0w
   0 0   0   0   -0w -0w
   0 0   0   0   0   -0w
   0 0   0   0   0   0  

The attention block is created as part of the first residual layer: :math:`y = x + attention(x;mask)`

::

   q)v:10; w:3; d:20; p:.1; h:4  / h-heads in multi-head attention

   q)q1:(`residual; seq(`sequential; (`selfattention;d;h;p;1b); (`drop;p)))

   q)q1
   `residual
   (`sequential;,(`selfattention;20;4;0.1;1b);,(`drop;0.1))

   q)-2 str q1:module q1;
   knn::Residual(
     (q1): torch::nn::Sequential(
       (0): knn::SelfAttention(dim=20, heads=4, dropout=0.1, norm=true)(
         (norm): torch::nn::LayerNorm([20], eps=1e-05, elementwise_affine=true)
         (in): torch::nn::Linear(in_features=20, out_features=60, bias=false)
         (drop): torch::nn::Dropout(p=0.1, inplace=false)
         (out): torch::nn::Linear(in_features=20, out_features=20, bias=true)
       )
       (1): torch::nn::Dropout(p=0.1, inplace=false)
     )
   )

Transformer block
^^^^^^^^^^^^^^^^^

The full transformer block is created by adding the residual layer with the self-attention to a second residual layer that consists of two linear layers with a `gelu <https://pytorch.org/docs/stable/generated/torch.nn.GELU.html>`_ activation in between:

::

   q)d:20; p:.1

   q)q2:seq(`sequential; (`layernorm;d);(`linear;d;d*4;0b); `gelu; (`linear;d*4;d); (`drop;p))

   q)q2
   `sequential
   ,(`layernorm;20)
   ,(`linear;20;80;0b)
   ,`gelu
   ,(`linear;80;20)
   ,(`drop;0.1)

   q)q2:module (`residual; q2)

   q)-2 str q2;
   knn::Residual(
     (q1): torch::nn::Sequential(
       (0): torch::nn::LayerNorm([20], eps=1e-05, elementwise_affine=true)
       (1): torch::nn::Linear(in_features=20, out_features=80, bias=false)
       (2): torch::nn::GELU()
       (3): torch::nn::Linear(in_features=80, out_features=20, bias=true)
       (4): torch::nn::Dropout(p=0.1, inplace=false)
     )
   )

The two residual layers together create the transformer block that is repeated for the GPT model used in the example scripts: the math sequence uses a shalllow network of 2 blocks and the deeper character-level language model uses 8.

::

   q)v:10; d:20; p:.1; h:4  / h-heads in multi-head attention

   q)q1:seq(`sequential; (`selfattention;d;h;p;1b); (`drop;p))
   q)q2:seq(`sequential; (`layernorm;d);(`linear;d;d*4;0b); `gelu; (`linear;d*4;d); (`drop;p))

Build one sequential block made of 2 residual layers:

::

   q)b:(`sequential; (`residual; q1); (`residual; q2))

   q)b
   `sequential
   (`residual;(`sequential;,(`selfattention;20;4;0.1;1b);,(`drop;0.1)))
   (`residual;(`sequential;,(`layernorm;20);,(`linear;20;80;0b);,`gelu;,(`linear;80;20);,(`drop;0.1)))

   q)b:module b

This block is repeated to increase the depth of the GPT model. The simpler model in  
`math.q <https://github.com/ktorch/examples/blob/master/gpt/math.q>`_ uses 2 transformer blocks
while the character-level language model used in 
`char.q <https://github.com/ktorch/examples/blob/master/gpt/char.q>`_ and
`callback.q <https://github.com/ktorch/examples/blob/master/gpt/callback.q>`_ uses 8 blocks.

The PyTorch representation of a single block:

::

   q)-2 str b;
   torch::nn::Sequential(
     (0): knn::Residual(
       (q1): torch::nn::Sequential(
         (0): knn::SelfAttention(dim=20, heads=4, dropout=0.1, norm=true)(
           (norm): torch::nn::LayerNorm([20], eps=1e-05, elementwise_affine=true)
           (in): torch::nn::Linear(in_features=20, out_features=60, bias=false)
           (drop): torch::nn::Dropout(p=0.1, inplace=false)
           (out): torch::nn::Linear(in_features=20, out_features=20, bias=true)
         )
         (1): torch::nn::Dropout(p=0.1, inplace=false)
       )
     )
     (1): knn::Residual(
       (q1): torch::nn::Sequential(
         (0): torch::nn::LayerNorm([20], eps=1e-05, elementwise_affine=true)
         (1): torch::nn::Linear(in_features=20, out_features=80, bias=false)
         (2): torch::nn::GELU()
         (3): torch::nn::Linear(in_features=80, out_features=20, bias=true)
         (4): torch::nn::Dropout(p=0.1, inplace=false)
       )
     )
   )

Decoder block
^^^^^^^^^^^^^
A normalization layer and a final linear layer at the end of the model maps from the embedding dimension to the vocabulary dimension,
i.e. to a tensor with a matrix for each observation in the batch with rows for each token in the input sequence
and columns with weights for all possible tokens.

::

   q)v:10; d:20  / for digits 0-9, vocabulary size is 10, embedding dim is 20 here
   q)q:seq(`sequential; (`layernorm;`norm;d); (`linear;`decode;d;v;0b))

   q)q
   `sequential
   ,(`layernorm;`norm;20)
   ,(`linear;`decode;20;10;0b)

   q)q:module q
   q)-2 str q;
   torch::nn::Sequential(
     (norm): torch::nn::LayerNorm([20], eps=1e-05, elementwise_affine=true)
     (decode): torch::nn::Linear(in_features=20, out_features=10, bias=false)
   )

   q)x:tensor(`randn; 3 6 20)
   q)y:forward(q;x)
   q)size y
   3 6 10

Transform
^^^^^^^^^

The output of the model is
`transformed <https://ktorch.readthedocs.io/en/latest/kmodules.html#module-transform>`_ 
differently depending on whether the forward calculation is run in training or evaluation mode.

When training, the output is reshaped from a 3-d tensor to a matrix, merging the batch and sequence dimension, so that the matrix rows match the length of the targets, the actual next tokens used in the `cross entropy loss <https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html>`_ calculation.

When the model is run in evaluation mode, only the final row for each sequence -- the weights for the last predicted token -- are output.

::

   q)v:10
   q)t:seq(`sequential; (`reshape;-1,v))
   q)t
   `sequential
   ,(`reshape;-1 10)

   q)e:seq(`sequential; (`select;1;-1))
   q)e
   `sequential
   ,(`select;1;-1)

The training and evaluation mode transforms are defined together into a 
`transform <https://ktorch.readthedocs.io/en/latest/kmodules.html#module-transform>`_ layer:

::

   q)q:(`transform; t; e)
   q)q
   `transform
   (`sequential;,(`reshape;-1 10))
   (`sequential;,(`select;1;-1))

   q)q:module q

   q)-2 str q;
   Transform((
     (train): torch::nn::Sequential(
       (0): Reshape(size=-1 10)
     )
     (eval): torch::nn::Sequential(
       (0): knn::Select(dim=1,ind=-1)
     )
   )

Running the forward calculation through the transform layer in training and evaluation mode:

::

   q)x:tensor(`randn; 3 6 10)  /after final linear layer
   q)y:forward(q;x)            /run in training mode
   q)size y
   18 10

   q)evaluate(q;x) / 3 sequences in batch x 10 weights for final token
   -1.625   0.2733  -0.6978 -0.01309 -0.9281  1.112 -0.9552 -0.5621  -1.981    0.3046 
    1.241  -0.8384   0.2689  0.9342   0.3258  1.397 -1.846   0.6229  -0.1792  -1.147 
    1.487  -0.4018  -0.7549 -1.186   -1.116   1.033 -0.63   -1.156    0.5084  -0.2428


Math
****

The `math.q <https://github.com/ktorch/examples/blob/master/gpt/math.q>`_ script builds a model to "learn" addition by predicting integer patterns.

Dataset
^^^^^^^

The sequences processed by the GPT model are sums of 2-digit numbers: the model attempts to predict the 3-digit sum given the preceding sequences.

::

   q)a:2  /number of digits
   q)x:{i:til prd 2#x:prd x#10; j:i div x; k:i mod x; (j;k;j+k)}a
   q)x:flip raze vs'[(a+0 0 1)#'10;x]

   q)count x
   10000

   q)5 ? x
   9 4 7 3 1 6 7  / 94 + 73 = 167
   6 2 3 4 0 9 6
   1 4 9 2 1 0 6
   4 0 0 6 0 4 6
   3 4 1 6 0 5 0

The inputs and targets are the sequences of tokens,
with the targets forming the next integer in the sequence of inputs:

::

   q)sample:6 4 5 8 1 2 2

   q)([]sequence:`input`target; (-1_sample; 1_sample))
   sequence sample     
   --------------------
   input    6 4 5 8 1 2    / x: 64 + 58 = 12..
   target   4 5 8 1 2 2    / y:..4 + 58 = 122

Only the targets that make up the output sum are used to calculate the cross entropy loss; the remaining digits are assigned ``-100``, which, 
`by convention <https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html>`_, is ignored by the loss calculation.

::

   q)sample:6 4 5 8 1 2 2

   q)@[1_sample; til -1+2*a; :; -100]
   -100 -100 -100 1 2 2

Model
^^^^^

The `math.q <https://github.com/ktorch/examples/blob/master/gpt/math.q>`_ script builds a very small GPT model for a vocabulary of 10 "tokens", the digits ``0-9``, with an embedding dimension of 128 and 4 heads for the attention layer. There are 2 transformer blocks, 400,128 trainable parameters overall.
The full PyTorch representation of the k api model is available `here  <https://github.com/ktorch/examples/blob/master/gpt/out/math.txt>`_.

Training
^^^^^^^^

The GPT model to learn addition via integer sequences is trained with a batch size of 500 and takes a few seconds on a NVIDIA GeForce GTX 1080 Ti GPU to achieve 99.9% accuracy in about 50 epochs, 100% accuracy is usually achieved after 75 epochs.  With cpu-only, training time is around 30 seconds to a minute.

Example mismatches at 99.9% test accuracy:

::

   mismatches in train: 1, test: 1

   dataset a  b  predict actual ok
   -------------------------------
   train   79 18 197     97     0 
   test    4  89 193     93     0 

Some training logs of 2-digit and 3-digit runs (fewer epochs required because of the much larger dataset) are `here  <https://github.com/ktorch/examples/blob/master/gpt/out/math.log>`_.

Character-level language model
******************************

Dataset
^^^^^^^

The dataset is a 1 mb text file of Shakespeare plays (`source <https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt>`_) 
that is named `shakespeare.txt <https://github.com/ktorch/examples/blob/master/gpt/data/shakespeare.txt>`_ in the example scripts.

The file is read in as text, then mapped to integers that corresponding to the distinct list of characters in the file:

::

   q)t:` sv read0`:data/shakespeare.txt

   q)char:asc distinct t
   q)char
   `s#"\n !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

   q)char:(`s#get enum:char!til count char)!char:asc distinct t

   q)char
   0 | 

   1 |  
   2 | !
   3 | $
   4 | &
   5 | '
   6 | ,
   7 | -
   8 | .
   9 | 3
   10| :
   11| ;
   12| ?
   13| A
   14| B
   15| C
   ..

   q)enum

   | 0
    | 1
   !| 2
   $| 3
   &| 4
   '| 5
   ,| 6
   -| 7
   .| 8
   3| 9
   :| 10
   ;| 11
   ?| 12
   A| 13
   B| 14
   C| 15
   ..

   q)t:enum t  /map chars to numbers
   q)t
   18 47 56 57 58 1 15 47 58 47 64 43 52 10 0 14 43 44 53 56 43 1 61 43 1 54 56 ..

The data is organized into batches of 200 sequences of 128 characters each:

::

   q)w:200; n:128
   q)i:(0N,w)#neg[i]?i:count[t]-1+n;  /rows: w starting indices of sequences length n+1
   q)b:i 0  /first batch
   q)x:t b+\:til n+1; /make w sequences of n+1 length;

The inputs are the sequences except for the final character; the targets are the next characters:

::

   q)char -1_first x  / inputs
   "on, not replying, yielded\nTo bear the golden yoke of sovereignty,\nWhich fo..

   q)char 1_first x  / targets
   "n, not replying, yielded\nTo bear the golden yoke of sovereignty,\nWhich fon..

Model
^^^^^

The `char.q <https://github.com/ktorch/examples/blob/master/gpt/char.q>`_ script builds a small GPT model for a vocabulary of 65 tokens, the characters encountered in the Shakespeare text file,
with an embedding dimension of 512 and 8 heads for the attention layer. There are 8 transformer blocks and 25 million trainable parameters overall.
The full PyTorch representation of the k api model is shown `here  <https://github.com/ktorch/examples/blob/master/gpt/out/char.txt>`_.

Training
^^^^^^^^

Training the character-level model takes about an hour per epoch on a NVIDIA GeForce GTX 1080 Ti and creates recognizable sequences after 2 passes through the data. CPU-only training takes 15-16 times longer, or about 32 hours for two epochs.

Batch size is set at 200 sequences at a time. A larger batch size of 256 uses more than the 11g of the GTX 1080's available memory,
e.g.

::

   'CUDA out of memory. Tried to allocate 128.00 MiB (GPU 0; 10.91 GiB total capacity; 9.93 GiB already allocated; 41.25 MiB free; 9.99 GiB reserved in total by PyTorch) If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF
     [4]  /home/t/examples/gpt/char.q:73: iter:
    nograd m;                                 /set gradients to undefined tensor
    s[`l]:backward(m; (-1_'x;u); raze 1_'x);  /calculate model output,loss & gradients
       ^

A training log of the character-level GPT model is `here  <https://github.com/ktorch/examples/blob/master/gpt/out/char.log>`_.

Generating sequences
^^^^^^^^^^^^^^^^^^^^

Once the GPT model has been trained, it is run in evaluation mode to generate the weights that are the basis for selecting the next character in the sequence. 
The raw weights from the model can be divided by a ``temperature`` factor that will smooth out the relative differences and make those tokens with lower relative weights more likely to be chosen.  
There is also an option to restrict the choices to only the `top k <https://pytorch.org/docs/stable/generated/torch.topk.html>`_ values,
along with a flag to take the token with the largest weight or sample from a `multinomial <https://pytorch.org/docs/stable/generated/torch.multinomial.html?highlight=multinomial>`_ distribution of the output probabilities.

Temperature is usually a scaling factor from 1.0 to 3.0, with the higher values in the range smoothing the relative differences between the weights after the `softmax <https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html>`_ is calculated:

::
  
   q){y!softmax x%/:y}[.3 -.5 .4; 1 1.5 2 3]
   1  | 0.3915 0.1759 0.4326
   1.5| 0.3766 0.2209 0.4025
   2  | 0.3674 0.2463 0.3863
   3  | 0.3572 0.2736 0.3693

The `topk <https://pytorch.org/docs/stable/generated/torch.topk.html>`_ function can be used to set all but the top k values to negative infinity so that the softmax assigns their probability to zero.

::

   q)x:normal 10#0e
   q)x
   -0.1238 -0.8407 -0.8363 2.158 -2.181 -0.5718 1.819 0.7981 -0.9481 -0.06846e

   q)topk(x;5)
   2.158 1.819 0.7981 -0.06846 -0.1238
   3     6     7      9        0      

   q)@[count[x]#max 0#x;k 1;:;first k:topk(x;5)]
   -0.1238 -0w -0w 2.158 -0w -0w 1.819 0.7981 -0w -0.06846e

   q)x:@[count[x]#max 0#x;k 1;:;first k:topk(x;5)]
   q)x:softmax x

   q)x
   0.04685 0 0 0.459 0 0 0.3268 0.1178 0 0.04952e

Given a set of probabilities for each possible token (optionally scaled by a temperature, and limited to the top k values),
the next token can be selected by sampling from the
`multinomial <https://pytorch.org/docs/stable/generated/torch.multinomial.html?highlight=multinomial>`_ distribution
defined by the probabilities, or by using the token with the highest probability.

::

   q)x
   0.04685 0 0 0.459 0 0 0.3268 0.1178 0 0.04952e

   q){(x key y)!get y}[x]count each group multinomial each 100#enlist x
   0.459  | 56
   0.3268 | 28
   0.1178 | 13
   0.04685| 2
   0.04952| 1

   q)argmax x  / index of largest probability
   3

All of the above choices are defined in the ``pick`` function in the script:

::

   q)pick
   {[t;k;s;x] /generate next char given temp, top k, sample flag & logits
    if[not t=1; x%:t];                                       /scale by temperature
    if[k; x:@[count[x]#max 0#x; j 1; : ;first j:topk(x;k)]]; /set -inf outside top k
    x:softmax x;                                             /output -> probabability
    $[s; multinomial x; argmax x]}

Callback module
***************

The `callback.q <https://github.com/ktorch/examples/blob/master/gpt/callback.q>`_ script fits the same model as
the `char.q <https://github.com/ktorch/examples/blob/master/gpt/char.q>`_ script but uses
a `callback <https://ktorch.readthedocs.io/en/latest/kmodules.html#module-callback>`_ module to do the forward calculation on
the embedding of the sequence, the list of transformer blocks and the final decode and transforms:

::

   q)v:65; w:200; d:512; h:8; n:128; p:.1
   q)emb:seq(`sequential`embed; (`embedseq;v;d;n); (`drop;p))
   q)emb
   `sequential`embed
   ,(`embedseq;65;512;128)
   ,(`drop;0.1)

The blocks are created as children of a `modulelist <https://pytorch.org/docs/stable/generated/torch.nn.ModuleList.html>`_:

::

   q)q1:seq(`sequential; (`selfattention;d;h;p;1b); (`drop;p))
   q)q2:seq(`sequential; (`layernorm;d);(`linear;d;d*4); `gelu; (`linear;d*4;d); (`drop;p))

   q)blk:(`sequential; (`residual;q1); (`residual;q2))
   q)blk
   `sequential
   (`residual;(`sequential;,(`selfattention;512;8;0.1;1b);,(`drop;0.1)))
   (`residual;(`sequential;,(`layernorm;512);,(`linear;512;2048);,`gelu;,(`linea..

   q)blk:enlist[`modulelist`blocks],8#enlist blk

   q)blk
   `modulelist`blocks
   (`sequential;(`residual;(`sequential;,(`selfattention;512;8;0.1;1b);,(`drop;0..
   (`sequential;(`residual;(`sequential;,(`selfattention;512;8;0.1;1b);,(`drop;0..
   (`sequential;(`residual;(`sequential;,(`selfattention;512;8;0.1;1b);,(`drop;0..
   (`sequential;(`residual;(`sequential;,(`selfattention;512;8;0.1;1b);,(`drop;0..
   (`sequential;(`residual;(`sequential;,(`selfattention;512;8;0.1;1b);,(`drop;0..
   (`sequential;(`residual;(`sequential;,(`selfattention;512;8;0.1;1b);,(`drop;0..
   (`sequential;(`residual;(`sequential;,(`selfattention;512;8;0.1;1b);,(`drop;0..
   (`sequential;(`residual;(`sequential;,(`selfattention;512;8;0.1;1b);,(`drop;0..

The end block is the last child of the callback module, adding a normalization layer and a linear layer to map from the embedding dimension to the number of possible tokens. 
The final `transform <https://ktorch.readthedocs.io/en/latest/kmodules.html#module-transform>`_ layer reshapes the output in training mode, merging the batch and sequence dimension into a single matrix with rows to match the number of elements in the targets used in the cross entropy loss calculation.  In evaluation mode, only the final row in each sequence is returned.

::

   q)end:seq(`sequential`end; (`layernorm;`norm;d); (`linear;`decode;d;v;0b))
   q)q1:seq(`sequential; (`reshape;-1,v))
   q)q2:seq(`sequential; (`select;1;-1))
   q)end,:enlist(`transform; q1; q2)

   q)end
   `sequential`end
   ,(`layernorm;`norm;512)
   ,(`linear;`decode;512;65;0b)
   (`transform;(`sequential;,(`reshape;-1 65));(`sequential;,(`select;1;-1)))

The three child layers are defined as part of the 
`callback <https://ktorch.readthedocs.io/en/latest/kmodules.html#module-callback>`_ parent:

::

   q)q:((`callback;`cb;`fwd;`tensor`tensor); emb; blk; end)

   q)q
   (`callback;`cb;`fwd;`tensor`tensor)
   (`sequential`embed;,(`embedseq;65;512;128);,(`drop;0.1))
   (`modulelist`blocks;(`sequential;(`residual;(`sequential;,(`selfattention;512..
   (`sequential`end;,(`layernorm;`norm;512);,(`linear;`decode;512;65;0b);(`trans..

   q)q:module q

   q)childnames q
   `embed`blocks`end

The full PyTorch representation of the k api model built as a ``callback`` module is shown `here  <https://github.com/ktorch/examples/blob/master/gpt/out/callback.txt>`_.

Forward call
^^^^^^^^^^^^

The options defined for the ``callback`` module indicate the callback function that will be called from c++ when a 
`forward or evaluate <https://ktorch.readthedocs.io/en/latest/modules.html#forward>`_ call is made:

::

   q)options q
   fn     | `fwd
   in     | `tensor`tensor
   out    | `tensor
   parms  | (`symbol$())!()
   buffers| (`symbol$())!()

   q)fwd
   {[m;x;y]
    x:kforward(m;`embed;x);
    s:(` sv`blocks,)each childnames(m;`blocks);
    use[x]{[m;y;x;c]use[x]kforward(m;c;x;y); x}[m;y]/[x;s];
    use[x]kforward(m;`end;x);
    x}

The ``fwd`` function first provides the input to the embedding layer for both the token and positional encoding:

::

    x:kforward(m;`embed;x);

Then the callback function processes each transformer block, taking the output from the previous layer as input, together
with the self-attention mask.

::

    s:(` sv`blocks,)each childnames(m;`blocks);
    use[x]{[m;y;x;c]use[x]kforward(m;c;x;y); x}[m;y]/[x;s];

Finally, the last decoding layer and transform:

::

   use[x]kforward(m;`end;x)


All the forward calls within the callback function use the
`kforward <https://ktorch.readthedocs.io/en/latest/kmodules.html#kforward>`_ utility,
which runs the forward calculation in training or test mode, with or without gradient calculation,
depending on the mode of the higher-level calling function, 
`forward, eforward or evaluate <https://ktorch.readthedocs.io/en/latest/modules.html#forward>`_,
which triggered the callback.
