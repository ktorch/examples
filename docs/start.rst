
.. _start:

Start
=====

A quick example, `spirals.q <https://github.com/ktorch/examples/blob/master/start/spirals.q>`_, demonstrates the main parts of a neural network and how the k api puts the pieces together.

The script generates some spirals with random noise for 3 sets of data:

::

   q)seed 1234
   q)n:1000; k:3                      / 1000 samples, 3 classes
   q)z:tensor(`randn; k,n; `double)   / create 1000 unit normals for each class

   q)(avg;sdev)@\:raze tensor z       / retrieve tensor, check characteristics
   0.005827275 1.007676

Define a function to generate spirals and add noise:

::

   q)spiral:{x:til[x]%x-1; flip x*/:(sin;cos)@\:z+sum 4*(x;y)}
   q)grid:{g:10 10; ./[(1+g+g)#0N;g{"j"$x+x*y}/:x;:;y]}   / display 21 x 21 grid

   q)x:raze spiral[n]'[til k;.2*tensor z]  /generate spirals w'noise
   q)y:til[k]where k#n                     /classes 0,1,2

   q)10?([]x;y)
   x                     y
   -----------------------
   -0.5926608 0.7513453  2
   -0.2018813 0.3271013  1
   -0.2313173 0.1600358  1
   0.8549625  -0.2202754 1
   0.4390136  -0.3875274 0
   -0.7305902 0.0408573  2
   0.08890049 -0.1057301 2
   -0.5851596 0.707221   2
   -0.2928673 0.3959515  1
   0.9519487  -0.2490344 1

   q)grid[x]y
                                        
            0       2   2 2 2 2 2          
          0 0       2 2 2 2 2 2 2 2        
        0 0     2 2 2 2 2 2 2 2 2 2 2      
      0 0 0   2 2 2 2             2 2 2    
    0 0 0     2 2 2         1       2 2 2  
    0 0 0   2 2 2 2     1 1 1         2 2  
    0 0 0   2 2 2     1 1 1 1 1 1       2 2
    0 0 0 2 2 2 2   1 1 1 1 1 1 1 1       2
    0 0 0   2 2 2   1 1 1     1 1 1 1      
    0 0 0     2 2 2 2 2 0 0     1 1 1      
    0 0 0     2 2 2 2 2 0 0     1 1 1      
      0 0 0 0   2 2 2 0 0 0 0   1 1 1      
      0 0 0 0 0 0   0 0 0 0     1 1 1      
        0 0 0 0 0 0 0 0 0       1 1 1      
            0 0 0 0 0 0 0     1 1 1 1      
                  0           1 1 1        
                            1 1 1 1        
                    1   1 1 1 1 1          
            1 1 1 1 1 1 1 1 1 1            
                  1 1 1 1 1                


Tensor ``x`` and ``y`` are the inputs and targets.

A `sequential model <https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html>`_ is defined with two `linear <https://pytorch.org/docs/stable/generated/torch.nn.Linear.html>`_ modules and a `relu <https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html>`_ activation function in between.
Passing inputs to this model will return a tensor with one row per observation and columns of raw, unnormalized scores for each class.

::

   q)x:tensor "e"$x
   q)y:tensor y

   q)q:module seq(`sequential; (`linear; 2; 100); `relu; (`linear; 100; k))

   q)-2 str q;   /PyTorch's string represention of the sequential module
   torch::nn::Sequential(
     (0): torch::nn::Linear(in_features=2, out_features=100, bias=true)
     (1): torch::nn::ReLU()
     (2): torch::nn::Linear(in_features=100, out_features=3, bias=true)
   )

   q)r:forward(q;x)

   q)tensor r
   -0.02184684 0.03731724 -0.07134955
   -0.02199468 0.03733625 -0.071125  
   -0.02215196 0.03736981 -0.07088985
   -0.02239865 0.03754543 -0.07055666
   -0.02235188 0.03726457 -0.07054836
   ..

These scores are given to the `cross entropy loss function <https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html>`_ along with the actual classification, 0, 1 or 2.

::

   q)l:loss`ce         / create a loss module
   q)a:loss(l;r;y)     / calculate loss on forward calculation
   q)tensor a          / loss is a scalar
   1.119298e
   q)backward a        / calculate the chain of gradients back through the network

The gradients are calculated and the linear weights are updated by an optimizer, in this case `simple gradient descent with momentum <https://pytorch.org/docs/stable/generated/torch.optim.SGD>`_.

::

  q)o:opt(`sgd; q; .2; .99) /gradient descent: learning rate .2, momentum .99

The main steps can be grouped in a function: zero out any previous gradients, perform forward calculation, then loss with forward result and target, followed by a backward calculation of the gradients and an optimization step to improve the linear weights.

::

   q)f:{[q;l;o;x;y]zerograd o; x:forward(q;x); backward y:loss(l;x;y); free x; step o; return y}


   q)f[q;l;o;x;y]
   1.119298e

   q)\ts:100 r:f[q;l;o;x;y]   /repeat training steps 100 times
   140 1728
   q)r                        /check for decreasing error
   0.07197823e

Create another function to evaluate the model, comparing the column with the maximum raw score to the actual class:

::

   q)g:{[q;x;y]p:100*avg tensor[y]={x?'max flip x}tensor x:forward(q;x); free x; p}

   q)g[q;x;y]   / about 98% accuracy
   97.73333

   q)\ts:100 r:f[q;l;o;x;y]  / try another 100 training steps
   153 1728
   q)r
   0.01378922e

   q)g[q;x;y]
   99.66667

The full run of the script creates a new data set with different random variables to test the fitted model:

::

   > q examples/start/spirals.q
   KDB+ 4.0 2020.05.04 Copyright (C) 1993-2020 Kx Systems
   l64/ 12(16)core 64037MB

                                        
            0       2   2 2 2 2 2          
          0 0       2 2 2 2 2 2 2 2        
        0 0     2 2 2 2 2 2 2 2 2 2 2      
      0 0 0   2 2 2 2             2 2 2    
    0 0 0     2 2 2         1       2 2 2  
    0 0 0   2 2 2 2     1 1 1         2 2  
    0 0 0   2 2 2     1 1 1 1 1 1       2 2
    0 0 0 2 2 2 2   1 1 1 1 1 1 1 1       2
    0 0 0   2 2 2   1 1 1     1 1 1 1      
    0 0 0     2 2 2 2 2 0 0     1 1 1      
    0 0 0     2 2 2 2 2 0 0     1 1 1      
      0 0 0 0   2 2 2 0 0 0 0   1 1 1      
      0 0 0 0 0 0   0 0 0 0     1 1 1      
        0 0 0 0 0 0 0 0 0       1 1 1      
            0 0 0 0 0 0 0     1 1 1 1      
                  0           1 1 1        
                            1 1 1 1        
                    1   1 1 1 1 1          
            1 1 1 1 1 1 1 1 1 1            
                  1 1 1 1 1                
   1083 1360
   Accuracy on test data: 99.93333%
   Accuracy w'new sample: 99.9%

Further examples will bring in larger datasets that are separated into batches and trained on smaller subsets of the data, but using the same basic set of steps for training and evaluating the models. 
Another version of the spirals example is in `run.q <https://github.com/ktorch/examples/blob/master/start/run.q>`_, which performs a similar optimization using higher-level training routines.
