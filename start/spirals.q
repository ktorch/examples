{key[x]set'x;}(`ktorch 2:`fns,1)[]

spiral:{x:til[x]%x-1; flip x*/:(sin;cos)@\:z+sum 4*(x;y)}
grid:{g:10 10; ./[(1+g+g)#0N;g{"j"$x+x*y}/:x;:;y]}

seed 1234
n:1000; k:3;                          /n:sample size, k:classes
z:tensor(`randn; k,n; `double)        /unit normal
x:raze spiral[n]'[til k;.2*tensor z]  /generate spirals w'noise
y:til[k]where k#n                     /class
show grid[x]y

x:tensor "e"$x
y:tensor y
q:module seq(`sequential; (`linear; 2; 100); `relu; (`linear; 100; k))
l:loss`ce               /cross-entropy loss
o:opt(`sgd; q; .2; .99) /gradient descent: learning rate .2, momentum .99

f:{[q;l;o;x;y]zerograd o; backward y:loss(l;x:forward(q;x);y); free'[(x;y)]; step o}
g:{[q;x;y]p:100*avg tensor[y]={x?'max flip x}tensor x:forward(q;x); free x; p}

\ts:1000 f[q;l;o;x;y]
-1"Accuracy on test data: ",string[g[q;x;y]],"%";

tensor(`randn; k,n; z)                               /new set of random variables
use[x]"e"$raze spiral[n]'[til k;.2*tensor z]         /rebuild spirals
-1"Accuracy w'new sample: ",string[g[q;x;y]],"%";    /check accuracy on data outside of fit
