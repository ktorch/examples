\l mnist.q
{key[x]set'get x}(`ktorch 2:`fns,1)[];  / define ktorch api fns

/ make labels longs, normalize images to mean 0, stddev 1
mnist:@[;`y`Y;"j"$]@[mnist;`x`X;%;255]
mnist:@[mnist;`x`X;{resize("e"$(z-x)%y;-1 1 28 28)}.(avg;dev)@\:mnist`x]

/ define sequential network layers, move to CUDA device if avail, add loss & optimizer
q:seq((`conv2d; 1;20;5);(`batchnorm;20);`relu;(`maxpool2d;2);
      (`conv2d;20;50;5);(`batchnorm;50);`relu;(`maxpool2d;2);`flatten;
      (`linear;800;500);`relu;(`linear;500;10))
to(q; c:device[])
m:model(q; loss`ce; opt(`sgd;q))

to(v:vector mnist`x`y; c)  / vector of training tensors, moved to gpu if available
to(V:vector mnist`X`Y; c)  / testing tensors used to evaluate model

/ set a learning rate schedule based on number of passes through the data
r:`s#0 5 10 15!.2 .1 .05 .02
msg:{-1 raze[("";"  lr:";"  loss:";"  test:";"  match:"),'.Q.fmt'[4 6 9 7 6;0 3 6 4 2;x]],"%";}
fit:{[m;v;V;w;r;i] lr(m;r@:i); a:avg[train(m;v;w)]; msg i,r,a,evalpct(m;V;1000); i+1}

\ts 20 fit[m;v;V;30;r]/0;  /20 passes, 30 images per batch
