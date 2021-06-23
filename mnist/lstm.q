p:first` vs hsym .z.f                      /path of this script
system"l ",1_string` sv p,`mnist.q         /load data w'mnist.q (assume same dir)
{key[x]set'get x}(`ktorch 2:`fns,1)[];     /define ktorch api fns

/ make labels longs, scale pixels from 0-255 to -1 to 1
mnist:@[;`y`Y;"j"$]@[mnist;`x`X;{resize("e"$-1+x%127.5;-1 28 28)}]

if[in[c:device[];cudadevices()]; setting`benchmark,1b] /set benchmark mode if CUDA

/ define sequential layers, move to CUDA device if available, add loss & optimizer
q:module`sequential
module(q; 1; (`lstm;     `recur; 28; 128; 2; 1b; 1b))
module(q; 1; (`lstmout;  `output))
module(q; 1; (`select;   `last; 1; -1))
module(q; 1; (`linear;   `decode; 128; 10))
to(q; c)
m:model(q; loss`ce; opt(`adamw;q))

to(v:vector mnist`x`y; c)  / vector of training tensors, moved to gpu if available
to(V:vector mnist`X`Y; c)  / testing tensors used to evaluate model

rate:{lr(x;r:.001 .0005 .0002 .0001 (y-1)mod 4); r} / cycle between learning rates .001 - .0001
msg:{-1 raze[("";"  lr:";"  loss:";"  test:";"  match:"),'.Q.fmt'[4 7 9 7 6;0 4 6 4 2;x]],"%";}
fit:{[m;v;V;w;i] r:rate[m]i; a:avg train(m;v;w); msg i,r,a,evaluate(m;V;1000;`loss`accuracy); i+1}
20 fit[m;v;V;60]/1;  /20 passes, 60 images per batch

/build table of mismatches in test dataset, convert to .png with labels
t:asc{([]y:x;yhat:y;ind:til count x)where not x=y}[mnist`Y]evaluate(m;V;1000;`max)
y:exec yhat by y from t                /mismatches by digit
x:exec ind by y from t                 /indices in test data
n:{max[x]-x}count'[x]                  /padding count to match max width (typically for '9')
x:"h"$127.5*1+mnist[`X]x               /wrongly classified images scaled to pixels 0-255
g:z,/:mnist[`n][y],'n#\:z:1 28 28#0h   /blank leading column, blanks to pad to same width
g:g,'mnist[`n][0N 1#key y],'x,'n#\:z   /join labels w'image list padded with blank images
g:makegrid(raze g; 2*count y; 1+max count'[y]; 2; 255)  / re-arrange into single grid of images

-2 "\nmismatches:"; show y
-2 "\ngrid of mismatches: ",1_string png(` sv p,`out`lstm.png;g);

\
Runtime on gpu: 57 seconds
After 100 trials, accuracy on test data:

low   | 99.09     
high  | 99.39     
median| 99.25     
stddev| 0.058
