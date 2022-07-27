p:first` vs hsym .z.f                      /path of this script
system"l ",1_string` sv p,`mnist.q         /load fns for reading MNIST (assume same dir)
{key[x]set'get x}(`ktorch 2:`fns,1)[];     /define ktorch api fns

/ make labels longs, scale pixels from 0-255 to -1 to 1
d:@[;`y`Y;"j"$]@[mnist[];`x`X;{resize("e"$-1+x%127.5;-1 784)}]

q:module enlist(`linear;784;10)
l:loss`ce
o:opt(`sgd;q;.04)
m:model(q;l;o)
train(m; `batchsize`shuffle; 100,1b)
train(m; d`x; d`y);

\ts:20 run m
-2 string[100*avg d.Y={x?max x}each evaluate(m;d`X)],"% test accuracy";
