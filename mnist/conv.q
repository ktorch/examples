\l mnist.q                              / read MNIST data
{key[x]set'get x}(`ktorch 2:`fns,1)[];  / define ktorch api fns

/ make labels longs, scale pixels from 0-255 to -1 to 1
mnist:@[;`y`Y;"j"$]@[mnist;`x`X;{resize("e"$-1+x%127.5;-1 1 28 28)}]

if[in[c:device[];cudadevices()]; setting`benchmark,1b]

/ define sequential network layers, move to CUDA device if avail, add loss & optimizer
q:seq((`conv2d; 1;20;5);`relu;`drop;(`maxpool2d;2);
      (`conv2d;20;50;5);`relu;`drop;(`maxpool2d;2);`flatten;
      (`linear;800;500);`relu;`drop;(`linear;500;10))
to(q; c)
m:model(q; loss`ce; opt(`sgd;q))

to(v:vector mnist`x`y; c)  / vector of training tensors, moved to gpu if available
to(V:vector mnist`X`Y; c)  / testing tensors used to evaluate model

/ set a learning rate schedule based on number of passes through the data
r:`s#0 6 11 16!.1 .05 .02 .01
msg:{-1 raze[("";"  lr:";"  loss:";"  test:";"  match:"),'.Q.fmt'[4 6 9 7 6;0 3 6 4 2;x]],"%";}
fit:{[m;v;V;w;r;i] lr(m;r@:i); a:avg[train(m;v;w)]; msg i,r,a,evalpct(m;V;1000); i+1}
20 fit[m;v;V;30;r]/1;  /20 passes, 30 images per batch

/build table of mismatches in test dataset, convert to .png with labels
r:asc{([]y:x;yhat:y;ind:til count x)where not x=y}[mnist`Y]last evalmax(m;V;1000)
y:exec yhat by y from r                  / mismatches by digit
x:exec ind by y from r                   / indices in test data
x:"h"$127.5*1+first''[mnist[`X]get x]    / get images, scale back to pixels from 0-255
p:{max[x]-x}get count each y             / padding count to match max width (typically for '9')
g:z,/:mnist[`n][y],'p#\:z:1 28 28#0h     / blank leading column, blanks to pad to same width
g:g,'mnist[`n][0N 1#key y],'x,'p#\:z     / join labels w'image list padded with blank images
g:makegrid(raze g; 2*count y; 1+max count'[y]; 2; 255)  / re-arrange into single grid of images

-2 "\nmismatches:"; show y
-2 "\ngrid of mismatches: ",1_string png(`:out/conv.png;g);
