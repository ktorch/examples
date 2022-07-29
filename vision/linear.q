{key[x]set'x}(`ktorch 2:`fns,1)[];   /load interface into root dir
p:first` vs hsym .z.f                /path of this script
system"l ",1_string` sv p,`cifar.q   /load CIFAR10 data w'script (assume in same dir)

/ simple linear model on flattened images: accuracy of around 50%
q:seq(`sequential;
      `flatten;
      (`linear; 32*32*3; 128);
      (`batchnorm1d; 128);
      `relu;
      (`linear; 128; 64);
      (`batchnorm1d; 64);
      `relu;
      (`linear;64;10))

q:module q
to(q;device[])
m:model(q; loss`ce; opt(`sgd;q;.002;.9))
d:@[;`y`Y;"j"$]@[cifar10[];`x`X;{"e"$-1+x%127.5}]

train(m; `batchsize`shuffle; 10,1b); train(m; d`x;d`y);
test(m;`batchsize`metrics!(1000;`accuracy)); test(m; d`X;d`Y);

\ts:20 {a:run x; a:a,testrun x; -2 raze("training loss: ";"  test accuracy: "),'string a;}m
