{key[x]set'x}(`ktorch 2:`fns,1)[];   /load interface into root dir
p:first` vs hsym .z.f                /path of this script
system"l ",1_string` sv p,`cifar.q   /load CIFAR10 data w'script (assume in same dir)
system"l ",1_string` sv p,`widenet.q /load wide resnet models

d:cifar10[]                          /read in CIFAR10 training & test data
d[`mean`std]:meanstd(d`x;0 2 3)      /calculate mean & stddev by RGB channel

/define transforms for training and test data:i
/random crop & horizontal flip for training, zscore for both
z:((`randomcrop;`crop;32;4);(`randomflip;`flip;.5;-1);(`zscore;`zscore;d`mean;d`std))
z:`transform,{seq`sequential,x}'[3 -1#\:z]

if[in[c:device[];cudadevices()]; setting`benchmark,1b] /set benchmark mode if CUDA
q:module widenet[16;8;.3;10]z  / use a small wide-resnet model
to(q; c)
m:model(q; loss`sce; opt(`sgd;q;.1;.9;0;.0005;1b)) / use smoothed cross entropy loss
to(v:vector d`x`y; c)  / vector of training tensors, moved to gpu if available
to(V:vector d`X`Y; c)  / testing tensors used to evaluate model

/ set learning rate schedule based on number of passes through the data (cosine annealing)
r:{(k+1)!.5*x*1+cos acos[-1]*(k:til y)%y}[.08]200
fmt:.Q.fmt'[4 9 9 7 6;0 6 6 4 2] /format epoch, rate, loss, test loss & accuracy
msg:{-1 raze[("";"  lr:";"  loss:";"  test:";"  accuracy:"),'fmt[x]],"%   ",string"v"$.z.T;}
fit:{[m;v;V;w;r;i] lr(m;r@:i); a:avg train(m;v;w); msg i,r,a,evaluate(m;V;1000;`loss`accuracy); i+1}
\ts count[r] fit[m;v;V;125;r]/1;  

/ build table of test results and top ten misclassifications
t:d[`s]@/:([]y:d`Y; yhat:evaluate(m;V;1000;`max))
show select[10;>n] n:count i by y,yhat from t where not y=yhat
