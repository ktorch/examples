/basic & bottleneck blocks, i:in, o:out channels, e:expansion, s:stride

basic:{[i;o;e;s] 
 seq((`sequential;`basic);
     (`conv2d;`conv1;i;o;3;s;1;`bias,0b); (`batchnorm2d;`bn1;o); (`relu;`relu1;1b);
     (`conv2d;`conv2;o;o;3;1;1;`bias,0b); (`batchnorm2d;`bn2;o))}

bottle:{[i;o;e;s]
 seq((`sequential;`bottleneck);
     (`conv2d;`conv1;i;o;1;1;  `bias,0b); (`batchnorm2d;`bn1;o); (`relu;`relu1;1b);
     (`conv2d;`conv2;o;o;3;s;1;`bias,0b); (`batchnorm2d;`bn2;o); (`relu;`relu2;1b);
     (`conv2d;`conv3;o;o*e;1;1;`bias,0b); (`batchnorm2d;`bn3;o*e))}

down:{[i;o;s] seq((`sequential;`downsample); (`conv2d;i;o;1;s;`bias,0b); (`batchnorm2d;o))}
downflag:{[i;o;e;s]not all(s=1;i=e*o)} /downsample true unless stride=1 & input=out channels*expansion

layer:{[e;n] /return matrices of in,out channels, expansion factor, stride & downsampling flag
 r:3{y[1]*x,2}[e]\64 64;                     /in,out channels for 4 resnet layers
 r:{@[(y,2)#z[1]*x,1;0;:;z]}[e]'[n;r];       /in,out for blocks within each layer
 r:r,''e;                                    /add expansion factor
 r:r,''r{y,(-1+count x)#1}'1,(-1+count r)#2; /stride for 1st blocks (stride=1 for subsequent blocks)
 r,''n{@[x#0b;0;:;downflag . first y]}'r}    /flag true if 1st block needs a downsampling layer

resid:{[f;i;o;e;s;b] /f: block fn, i:in, o:out channels, e:expansion, s:stride, b:true if 1st
 r:f[i;o;e;s];             /basic or bottleneck layer
 a:enlist(`relu;`relu;1b); /final activation
 `residual,$[b; (r;down[i;e*o;s];a); (r;a)]}

resnet:{[f;a;n;c]  /f:`basic/`bottle, a:alternate flag, n:number of blocks, c:classes
 q:seq((`sequential;`resnet); 
       (`conv2d;`conv1;3;64;3;1;1;`bias,0b);  /alternate convolution 3x3, stride=1, pad=1
       (`conv2d;`conv1;3;64;7;2;3;`bias,0b);  /published convolution 7x7, stride=2, pad=3
       (`batchnorm2d;`bn1;64);
       (`relu;`relu;1b);
       (`maxpool2d;`maxpool;3;2;1));          /alternate version skips max pooling
 q@:(0 2 3 4 5; 0 1 3 4)a;                    /define resnet as standard or alternate
 e:1 4`basic`bottle?f;
 r:enlist'[`seqnest,'`$"layer",/:string 1+til count n];
 q,:r,'resid[f]./:/:layer[e;n];
 q,enlist'[((`adaptavg2d;`avgpool;1 1); (`flatten;`flatten;1); (`linear;`fc;e*512;c))]}

resnet18:  resnet[`basic;  0b; 2 2  2 2]
resnet18a: resnet[`basic;  1b; 2 2  2 2]  / alternate (3-5% better w'cifar10 images)
resnet34:  resnet[`basic;  0b; 3 4  6 3]
resnet34a: resnet[`basic;  1b; 3 4  6 3]
resnet50:  resnet[`bottle; 0b; 3 4  6 3]
resnet101: resnet[`bottle; 0b; 3 4 23 3]
resnet152: resnet[`bottle; 1b; 3 8 36 3]
