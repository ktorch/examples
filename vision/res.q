{key[x]set'x}(`ktorch 2:`fns,1)[];
p:first` vs hsym .z.f                 /path of this script
system"l ",1_string` sv p,`cifar.q    /load CIFAR10 data w'script (assume in same dir)
system"l ",1_string` sv p,`resnet.q   /load resnet models

d:cifar10[]                          /read in CIFAR10 training & test data
d[`mean`std]:meanstd(d`x;0 2 3)      /calculate mean & stddev by RGB channel
@[`d;`x`X;{zscore(x;d`mean;d`std)}]; /standardize train & test data to mean zero, stddev of 1
q)d[`x],:Flip(d`x;-1)                /add horizontal flip of each training image
q)d[`y],:d`y                         /repeat training targets

if[in[c:device[];cudadevices()]; setting`benchmark,1b] /set benchmark mode if CUDA
q:module resnet18a 10  /create small resnet model
to(q;c)                /move to CUDA if available, add loss & optimizer
m:model(q; loss`ce; opt(`sgd;q;.1;.9;0;.0005;1b))
to(v:vector "ej"$d`x`y; c)  / vector of tensors of training data
to(V:vector "ej"$d`X`Y; c)  / testing tensors used to evaluate model

/ set learning rate schedule based on number of passes through the data (cosine annealing)
r:{(k+1)!.5*x*1+cos acos[-1]*(k:til y)%y}[.08]60
fmt:.Q.fmt'[4 9 9 7 6;0 6 6 4 2] /format epoch, rate, loss, test loss & accuracy
msg:{-1 raze[("";"  lr:";"  loss:";"  test:";"  accuracy:"),'fmt[x]],"%   ",string"v"$.z.T;}
fit:{[m;v;V;w;r;i] lr(m;r@:i); a:avg train(m;v;w); msg i,r,a,evaluate(m;V;1000;`loss`accuracy); i+1}
\ts count[r] fit[m;v;V;125;r]/1;  / test for no. of epochs in learning rate schedule, batch size 125

/ build table of test results
t:d[`s]@/:([]y:d`Y; yhat:evaluate(m;V;1000;`max))
show select[10;>n] n:count i by y,yhat from t where not y=yhat
