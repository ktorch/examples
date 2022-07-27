p:first` vs hsym .z.f                      /path of this script
system"l ",1_string` sv p,`mnist.q         /load fns for reading MNIST (assume same dir)
{key[x]set'get x}(`ktorch 2:`fns,1)[];     /define ktorch api fns

/ make labels longs, scale pixels from 0-255 to -1 to 1
d:@[;`y`Y;"j"$]@[mnist[];`x`X;{resize("e"$-1+x%127.5;-1 784)}]

if[in[c:device[];cudadevices()]; setting`benchmark,1b]

/ define sequential network layers, move to CUDA device if avail, add loss & optimizer
q:module seq(`sequential; (`linear;784;800); `relu; (`linear;800;10))
to(q; c)
m:model(q; loss`ce; opt(`sgd;q;.05;.9;`nesterov,1b)) 

train(m; `batchsize`shuffle; 100,1b);                train(m; d`x; d`y);
test(m; `batchsize`metrics; (1000;`loss`accuracy));   test(m; d`X; d`Y);

msg:{-1 raze[("";"  loss:";"  test:";"  accuracy:"),'.Q.fmt'[4 9 7 6;0 6 4 2;x]],"%";}
fit:{msg y,raze`run`testrun@\:x; y+1}
\ts 50 fit[m]/1;
