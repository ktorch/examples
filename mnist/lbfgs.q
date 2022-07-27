p:first` vs hsym .z.f                      /path of this script
system"l ",1_string` sv p,`mnist.q         /fns for reading MNIST files
{key[x]set'get x}(`ktorch 2:`fns,1)[];     /define ktorch api fns

/ make labels longs, scale pixels from 0-255 to -1 to 1
d:mnist` sv p,`data
d:@[;`y`Y;"j"$]@[d;`x`X;{resize("e"$-1+x%127.5;-1 1 28 28)}]

if[in[c:device[];cudadevices()]; setting`benchmark,1b] /set benchmark mode if CUDA

/ define sequential layers, move to CUDA device if available, add loss & optimizer
q: ((`conv2d; 1;20;5); `relu; `drop; (`maxpool2d;2))
q,:((`conv2d;20;50;5); `relu; `drop; (`maxpool2d;2); `flatten)
q,:((`linear;800;500); `relu; `drop; (`linear;500;10))
q:module seq `sequential,q
to(q; c)

m:model(q; loss(`ce;`smoothing,.01); opt(`lbfgs;q))
train(m; `batchsize`shuffle; 1000,1b);       train(m; d`x; d`y);
test(m;  `batchsize`metrics; 1000,`accuracy); test(m; d`X; d`Y);

/ set learning rate schedule based on number of passes through the data
fmt:.Q.fmt'[4 7 9 6;0 4 6 2]
msg:{-2 raze[("";"  lr:";"  training loss:";"  test accuracy:"),'fmt x],"%";}
fit:{[m;i] msg i,lr[m],raze`run`testrun@\:m; i+1}
\ts 50 fit[m]/1; 

/build table of mismatches in test dataset, convert to .png with labels
test(m;`metrics;`predict)
s:digits[] / define a set of standardized digits for labeling
t:asc{([]y:x;yhat:y;ind:til count x)where not x=y}[d`Y]testrun m
y:exec yhat by y from t            /mismatches by digit
x:exec ind by y from t             /indices in test data
n:{max[x]-x}count'[x]              /padding count to match max width (typically for '9')
x:"h"$127.5*1+first''[d[`X]get x]  /get images, scaled to pixels from 0-255
g:z,/:s[y],'n#\:z:1 28 28#0h       /blank leading column, blanks to pad to same width
g:g,'s[0N 1#key y],'x,'n#\:z       /join labels w'image list padded with blank images
g:makegrid(raze g; 2*count y; 1+max count'[y]; 2; 255)  /re-arrange into single grid of images

-2 "\nmismatches:"; show y
-2 "\ngrid of mismatches: ",1_string png(` sv p,`out`lbfgs.png;g);
