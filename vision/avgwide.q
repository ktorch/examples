{key[x]set'x}(`ktorch 2:`fns,1)[];   /load interface into root dir
p:first` vs hsym .z.f                /path of this script
system"l ",1_string` sv p,`cifar.q   /load CIFAR10 data w'script (assume in same dir)
system"l ",1_string` sv p,`widenet.q /load wide resnet models

d:cifar10[]                          /read in CIFAR10 training & test data
d[`mean`std]:meanstd(d`x;0 2 3)      /calculate mean & stddev by RGB channel

/define transforms for training and test data
/random crop & horizontal flip for training, zscore for both
z:((`randomcrop;`crop;32;4);(`randomflip;`flip;.5;-1);(`zscore;`zscore;d`mean;d`std))
z:`transform,{seq`sequential,x}'[3 -1#\:z]

if[in[c:device[];cudadevices()]; setting`benchmark,1b] /set benchmark mode if CUDA
q:module widenet[16;8;.3;10]z  / use a small wide-resnet model
to(q; c)

m:model(q;
        loss(`ce;`smoothing,.1);        / smoothed cross entropy loss
         opt(`sgd;q;.1;.9;0;.0005;1b))  / .9 momentum, .0005 decay

w:125 1000
train(m; `batchsize`shuffle; (w 0;1b)); n:train(m; d`x; "j"$d`y);
test(m; `batchsize`metrics; (w 1; `loss`accuracy)); n:n,test(m; d`X; "j"$d`Y)
-2 ", "sv("train: ";"test: "),'{raze string[(x;y)],'(" batches of ";" images")}'[n;w];

/ set learning rate schedule based on number of passes through the data (cosine annealing)
r:{(k+1)!.5*x*1+cos acos[-1]*(k:til y)%y}[.08]100
fmt:.Q.fmt'[4 9 9 7 6;0 6 6 4 2] /format epoch, rate, loss, test loss & accuracy
msg:{-1 raze[("";"  lr:";"  loss:";"  test:";"  accuracy:"),'fmt[x]],"%   ",string"v"$.z.T;}

fit:{[m;a;r;n;i]
 if[i=n; -2"\nweight averaging:"]; /indicate start of weight averaging
 lr(m;r@:i&n);                     /decay learning rate until averaging begins
 msg i,r,raze`run`testrun@\:m;     /report train & test steps
 $[i=n; use[a]copyparms m; i>n; avgparms(m;a)]; i+1} /initialize, then avg weights

a:dict(); n:{1+ceiling y-x*y}[.2]count r /initialize dictionary for averaging, use final 20% of iterations
count[r] fit[m;a;r;n]/1;
if[not all copyparms(m;a); '"unable to replace all weights with averages"];
batchnorm m; / recalculate running mean & variance for batchnorm layers

-2 "\ntest accuracy using average weights from epoch ",string[n],"-",string[count r],": ",.Q.fmt[6;2;last testrun m],"%";

/ build table of test results and top ten misclassifications
test(m;`metrics;`predict)
t:d[`s]@/:([]y:d`Y; yhat:testrun m)
show select[10;>n] n:count i by y,yhat from t where not y=yhat
