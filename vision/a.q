{key[x]set'x}(`ktorch 2:`fns,1)[];   /load interface into root dir
p:first` vs hsym .z.f                /path of this script
system"l ",1_string` sv p,`cifar.q   /load CIFAR10 data w'script (assume in same dir)

n:2048 512 128 64
q:seq(`sequential;
      `flatten;
      (`linear; 32*32*3; n 0); (`batchnorm1d; n 0); `relu;
      (`linear; n 0; n 1); (`batchnorm1d; n 1); `relu;
      (`linear; n 1; n 2); (`batchnorm1d; n 2); `relu;
      (`linear; n 2; n 3); (`batchnorm1d; n 3); `relu;
      (`linear; n 3;10))

q:module q
to(q;`cuda)
m:model(q; loss(`ce;`smoothing,.01); opt(`sgd;q;.1;.9))

d:cifar10[]
d:@[;`y`Y;"j"$]@[d;`x`X;{"e"$-1+x%127.5}]

train(m; `batchsize`shuffle`metrics; (100;1b;`loss`accuracy))
train(m; d`x; d`y)
test(m;`batchsize`metrics; 1000,`accuracy)
test(m; d`X; d`Y)

f:(-2 raze .Q.fmt'[4 7 7 7;0 2 2 2]@)
\ts:200 f (i+:1.0),raze`run`testrun@\:m

