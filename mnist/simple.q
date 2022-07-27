p:first` vs hsym .z.f                      /path of this script
system"l ",1_string` sv p,`mnist.q         /fns for reading MNIST files
{key[x]set'get x}(`ktorch 2:`fns,1)[];     /define ktorch api fns

/ make labels longs, scale pixels from 0-255 to -1 to 1
d:mnist[]; d:@[;`y`Y;"j"$]@[d;`x`X;{resize("e"$-1+x%127.5;-1 1 28 28)}]

if[in[c:device[];cudadevices()]; setting`benchmark,1b] /set benchmark mode if CUDA

/ input output size pool
q:( 1    66    3    0;
   66    64    3    0;
   64    64    3    0;
   64    64    3    0;
   64    96    3    2;
   96    96    3    0;
   96    96    3    0;
   96    96    3    0;
   96    96    3    0;
   96   144    3    2;
  144   144    1    0;
  144   178    1    0;
  178   216    3    7)

f:{[i;o;s;p]
 c:(`conv2d;i;o;s;1;`same);      /convolution layer w'padding=same
 b:(`batchnorm2d;o;1e-05;.05);   /batchnorm w'momentum of .95
 f:`relu; d:(`drop;.2); m:(`maxpool2d;p);
 / final global max pool needs reshape & linear layer
 if[p=7; r:(`reshape;-1,o); l:(`linear;o;10)];
 $[p=0; (c;b;f;d); p=2; (c;b;f;m;d); (c;b;f;m;r;d;l)]}

q:module `sequential,enlist each raze f ./:q
w:{x where x like "*.weight"}where parmtypes[q]=`conv2d
{xnormal(x;y;gain`relu)}[q]'[w]; /xavier initialization for conv layers
to(q; c)
 
m:model(q; loss(`ce;`smoothing,.01); opt(`adamw;q;`beta1`beta2`decay!.5 .95 .1))
train(m; `batchsize`shuffle;  100,1b);       train(m; d`x; d`y);
test(m;  `batchsize`metrics; 1000,`accuracy); test(m; d`X; d`Y);

/ set learning rate schedule based on number of passes through the data
r:{(k+1)!y|.5*x*1+cos acos[-1]*(k:til z)%z}[.01;.0]50
fmt:.Q.fmt'[4 7 9 6;0 4 6 2]
msg:{-2 raze[("";"  lr:";"  training loss:";"  test accuracy:"),'fmt x],"%";}
fit:{[m;r;i] lr(m;r@:i); msg i,r,raze`run`testrun@\:m; i+1}

n:count[r],{y,x div y}[count d`x;train(m;`batchsize)]
-2 raze("epochs: "; ", batch size: ";", iterations per epoch: "),'string n;
\ts count[r] fit[m;r]/1; 

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
-2 "\ngrid of mismatches: ",1_string png(` sv p,`out`simple.png;g);
