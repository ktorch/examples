p:first` vs hsym .z.f                      /path of this script
system"l ",1_string` sv p,`mnist.q         /load data w'mnist.q (assume same dir)
{key[x]set'get x}(`ktorch 2:`fns,1)[];     /define ktorch api fns

/ make labels longs, scale pixels from 0-255 to -1 to 1
d:mnist[]; d:@[;`y`Y;"j"$]@[d;`x`X;{resize("e"$-1+x%127.5;-1 28 28)}]

if[in[c:device[];cudadevices()]; setting`benchmark,1b] /set benchmark mode if CUDA

/ define sequential layers, move to CUDA device if available, add loss & optimizer
q:module`recur
module(q; 1; (`lstm;     `lstm; 28; 128; 2; 1b; 1b))
module(q; 1;  `sequential);
module(q; 2; (`select;   `last; 1; -1))
module(q; 2; (`linear;   `decode; 128; 10))
to(q; c)
m:model(q; loss`ce; opt(`adamw;q))

train(m; `batchsize`hidden`sync; (60;0b;1b));  train(m; d`x; d`y);
test(m;  `batchsize`metrics; (1000;`accuracy)); test(m; d`X; d`Y);

rate:{lr(x;r:.001 .0005 .0002 .0001 (y-1)mod 4); r} / cycle between learning rates .001 - .0001
msg:{-1 raze[("";"  lr:";"  training loss:";"  test accuracy:"),'.Q.fmt'[4 7 9 6;0 4 6 2;x]],"%";}
fit:{[m;d;i] r:rate[m]i; a:run m; msg i,r,a,testrun m; i+1}
\ts 40 fit[m;d]/1;  /40 passes through the data

/build table of mismatches in test dataset, convert to .png with labels
test(m;`metrics;`predict)
s:digits[] / define a set of standardized digits for labeling
t:asc{([]y:x;yhat:y;ind:til count x)where not x=y}[d`Y]testrun m;
y:exec yhat by y from t        /mismatches by digit
x:exec ind by y from t         /indices in test data
n:{max[x]-x}count'[x]          /padding count to match max width (typically for '9')
x:"h"$127.5*1+d[`X]x           /wrongly classified images scaled to pixels 0-255
g:z,/:s[y],'n#\:z:1 28 28#0h   /blank leading column, blanks to pad to same width
g:g,'s[0N 1#key y],'x,'n#\:z   /join labels w'image list padded with blank images
g:makegrid(raze g; 2*count y; 1+max count'[y]; 2; 255)  / re-arrange into single grid of images

-2 "\nmismatches:"; show y
-2 "\ngrid of mismatches: ",1_string png(` sv p,`out`lstm.png;g);
