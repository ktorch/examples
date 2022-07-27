{key[x]set'x}(`ktorch 2:`fns,1)[];   /load interface into root dir
p:first` vs hsym .z.f                /path of this script
system"l ",1_string` sv p,`cifar.q   /load CIFAR10 data w'script (assume in same dir)

seed 42

q:seq(`sequential;
      `flatten;
      (`linear; 32*32*3; 64);
      (`batchnorm1d; 64);
      `relu;
      (`linear; 64; 32);
      (`batchnorm1d; 32);
      `relu;
      (`linear;32;10))

q:module q
m:model(q; loss`ce; opt(`adam;q;.0004))

d:cifar10[]

train(m; `batchsize`shuffle; 10,1b)
train(m; "e"$d`x; "j"$d`y)

\ts:5 show run m
