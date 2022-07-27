/https://github.com/locuslab/convmixer-cifar10
{key[x]set'x}(`ktorch 2:`fns,1)[];   /load interface into root dir
p:first` vs hsym .z.f                /path of this script
system"l ",1_string` sv p,`cifar.q   /load CIFAR10 data w'script (assume in same dir)
d:cifar10[]                          /read in CIFAR10 training & test data
d[`mean`std]:meanstd(d`x;0 2 3)      /calculate mean & stddev by RGB channel

h:256; k:5; s:2; n:10 /hidden dim, convolution kernel size & stride, number of classes

q:((`randomcrop;`crop;32;4);        /random crop if training
   (`randomflip;`flip;.5;-1);       /random flip if training
   (`zscore;`zscore;d`mean;d`std))  /zscore for train & eval
q:(`sequential; `transform,`sequential,'3 -1#\:enlist'[q])

q,:enlist each ((`conv2d;3;h;s;s); `gelu; (`batchnorm2d;h))  /initial convolution layers

r:((`residual; (`sequential; enlist(`conv2d;h;h;k;`groups`pad!h,`same); `gelu; enlist(`batchnorm2d;h)));
    enlist(`conv2d;h;h;1);
    enlist `gelu;
    enlist(`batchnorm2d;h));

q:8 {y,x}[r]/q  / add 8 residual blocks
q,:enlist each( (`adaptavg2d;1 1); `flatten; (`linear;h;n))
q:module q

if[in[c:device[];cudadevices()]; setting`benchmark,1b] /set benchmark mode if CUDA device
to(q;c)                                                /move module to CUDA if available
m:model(q; loss`ce; opt(`lamb;q;`decay,.001))          /make model w'lamb optimizer

w:500 1000 
train(m; `batchsize`shuffle; (w 0;1b)); n:train(m; d`x; "j"$d`y);
test(m; `batchsize`metrics; (w 1; `loss`accuracy)); n:n,test(m; d`X; "j"$d`Y)
-2 ", "sv("train: ";"test: "),'{raze string[(x;y)],'(" batches of ";" images")}'[n;w];

/ set learning rate schedule based on number of passes through the data (cosine annealing)
r:{(k+1)!.5*x*1+cos acos[-1]*(k:til y)%y}[.02]40
fmt:.Q.fmt'[4 9 9 7 6;0 6 6 4 2] /format epoch, rate, loss, test loss & accuracy
msg:{-1 raze[("";"  lr:";"  loss:";"  test:";"  accuracy:"),'fmt[x]],"%   ",string"v"$.z.T;}
fit:{[m;r;i] lr(m;r@:i); msg i,r,raze`run`testrun@\:m; i+1} 

count[r] fit[m;r]/1;

/ build table of test results and top ten misclassifications
test(m;`metrics;`predict)
t:d[`s]@/:([]y:d`Y; yhat:testrun m)
show select[10;>n] n:count i by y,yhat from t where not y=yhat

\
train: 100 batches of 500 images, test: 10 batches of 1000 images
  1.  lr: 0.020000  loss: 1.352406  test: 1.5130  accuracy: 54.90%   08:51:41
  2.  lr: 0.019969  loss: 0.896929  test: 0.9344  accuracy: 69.13%   08:52:15
  3.  lr: 0.019877  loss: 0.730319  test: 1.0696  accuracy: 68.80%   08:52:50
  4.  lr: 0.019724  loss: 0.635800  test: 0.7149  accuracy: 77.13%   08:53:25
  5.  lr: 0.019511  loss: 0.561631  test: 0.7062  accuracy: 78.14%   08:53:59
  6.  lr: 0.019239  loss: 0.497103  test: 0.8235  accuracy: 74.40%   08:54:34
  7.  lr: 0.018910  loss: 0.435854  test: 0.6193  accuracy: 80.37%   08:55:10
  8.  lr: 0.018526  loss: 0.388576  test: 0.5292  accuracy: 83.04%   08:55:45
  9.  lr: 0.018090  loss: 0.348940  test: 0.4979  accuracy: 83.92%   08:56:20
 10.  lr: 0.017604  loss: 0.313513  test: 0.5463  accuracy: 82.47%   08:56:55
 11.  lr: 0.017071  loss: 0.281312  test: 0.5407  accuracy: 83.87%   08:57:30
 12.  lr: 0.016494  loss: 0.258607  test: 0.4668  accuracy: 85.61%   08:58:06
 13.  lr: 0.015878  loss: 0.233144  test: 0.5306  accuracy: 84.70%   08:58:41
 14.  lr: 0.015225  loss: 0.210563  test: 0.4838  accuracy: 85.50%   08:59:16
 15.  lr: 0.014540  loss: 0.189287  test: 0.4158  accuracy: 87.44%   08:59:51
 16.  lr: 0.013827  loss: 0.167971  test: 0.5082  accuracy: 86.43%   09:00:27
 17.  lr: 0.013090  loss: 0.152949  test: 0.4346  accuracy: 88.17%   09:01:02
 18.  lr: 0.012334  loss: 0.138633  test: 0.4622  accuracy: 87.47%   09:01:37
 19.  lr: 0.011564  loss: 0.120079  test: 0.4204  accuracy: 88.61%   09:02:12
 20.  lr: 0.010785  loss: 0.108879  test: 0.4736  accuracy: 88.20%   09:02:48
 21.  lr: 0.010000  loss: 0.097000  test: 0.4916  accuracy: 87.50%   09:03:23
 22.  lr: 0.009215  loss: 0.081201  test: 0.4713  accuracy: 89.00%   09:03:58
 23.  lr: 0.008436  loss: 0.073730  test: 0.4784  accuracy: 88.93%   09:04:33
 24.  lr: 0.007666  loss: 0.062277  test: 0.5224  accuracy: 88.59%   09:05:08
 25.  lr: 0.006910  loss: 0.054226  test: 0.5537  accuracy: 88.74%   09:05:43
 26.  lr: 0.006173  loss: 0.045954  test: 0.5783  accuracy: 88.86%   09:06:19
 27.  lr: 0.005460  loss: 0.039732  test: 0.4994  accuracy: 90.11%   09:06:54
 28.  lr: 0.004775  loss: 0.031847  test: 0.5375  accuracy: 89.67%   09:07:29
 29.  lr: 0.004122  loss: 0.027166  test: 0.5371  accuracy: 89.82%   09:08:04
 30.  lr: 0.003506  loss: 0.019995  test: 0.5510  accuracy: 90.33%   09:08:39
 31.  lr: 0.002929  loss: 0.018690  test: 0.5670  accuracy: 90.26%   09:09:15
 32.  lr: 0.002396  loss: 0.014606  test: 0.5782  accuracy: 90.45%   09:09:50
 33.  lr: 0.001910  loss: 0.012111  test: 0.5761  accuracy: 90.48%   09:10:25
 34.  lr: 0.001474  loss: 0.011093  test: 0.5441  accuracy: 90.90%   09:11:00
 35.  lr: 0.001090  loss: 0.007841  test: 0.5814  accuracy: 90.87%   09:11:35
 36.  lr: 0.000761  loss: 0.006905  test: 0.6031  accuracy: 90.73%   09:12:10
 37.  lr: 0.000489  loss: 0.005088  test: 0.5911  accuracy: 91.12%   09:12:45
 38.  lr: 0.000276  loss: 0.005289  test: 0.5797  accuracy: 91.14%   09:13:20
 39.  lr: 0.000123  loss: 0.004278  test: 0.6090  accuracy: 90.92%   09:13:56
 40.  lr: 0.000031  loss: 0.005303  test: 0.6060  accuracy: 91.02%   09:14:31

y          yhat    | n 
-------------------| --
dog        cat     | 87
cat        dog     | 64
automobile truck   | 34
cat        bird    | 31
ship       airplane| 30
bird       deer    | 30
cat        deer    | 26
airplane   bird    | 26
deer       bird    | 24
horse      dog     | 22

