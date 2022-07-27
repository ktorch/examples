{key[x]set'x}(`ktorch 2:`fns,1)[];
p:first` vs hsym .z.f                 /path of this script
system"l ",1_string` sv p,`cifar.q    /load CIFAR10 data w'script (assume in same dir)
system"l ",1_string` sv p,`resnet.q   /load resnet models

d:cifar10[]                          /read in CIFAR10 training & test data
d[`mean`std]:meanstd(d`x;0 2 3)      /calculate mean & stddev by RGB channel
@[`d;`x`X;{zscore(x;d`mean;d`std)}]; /standardize train & test data to mean zero, stddev of 1
d[`x],:Flip(d`x;-1)                  /add horizontal flip of each training image
d[`y],:d`y                           /repeat training targets

if[in[c:device[];cudadevices()]; setting`benchmark,1b] /set benchmark mode if CUDA
if[in[c:`cuda:1; cudadevices()]; setting`benchmark,1b] /set benchmark mode if CUDA
q:module resnet18a 10  /create small resnet model
to(q;c)                /move to CUDA if available, add loss & optimizer
m:model(q; loss`ce; opt(`sgd;q;.1;.9;0;.0005;1b))

w:125 1000
train(m; `batchsize`shuffle; (w 0;1b)); n:train(m; d`x; "j"$d`y);
test(m; `batchsize`metrics; (w 1; `loss`accuracy)); n:n,test(m; d`X; "j"$d`Y)
-2 ", "sv("train: ";"test: "),'{raze string[(x;y)],'(" batches of ";" images")}'[n;w];

/ set learning rate schedule based on number of passes through the data (cosine annealing)
r:{(k+1)!.5*x*1+cos acos[-1]*(k:til y)%y}[.08]60
fmt:.Q.fmt'[4 9 9 7 6;0 6 6 4 2] /format epoch, rate, loss, test loss & accuracy
msg:{-1 raze[("";"  lr:";"  loss:";"  test:";"  accuracy:"),'fmt[x]],"%   ",string"v"$.z.T;}
fit:{[m;r;i] lr(m;r@:i); msg i,r,raze`run`testrun@\:m; i+1}
\ts count[r] fit[m;r]/1;  / test for no. of epochs in learning rate schedule

/ build table of test results, select top ten misclassifications
test(m;`metrics;`predict)
t:d[`s]@/:([]y:d`Y; yhat:testrun m)
show select[10;>n] n:count i by y,yhat from t where not y=yhat
