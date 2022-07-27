{key[x]set'x;}(`ktorch 2:`fns,1)[]

spiral:{x:til[x]%x-1; flip"e"$x*/:(sin;cos)@\:z+sum 4*(x;y)}
grid:{g:10 10; ./[(1+g+g)#0N;g{"j"$x+x*y}/:x;:;y]}

n:1000; k:3;                          /n:sample size, k:classes
z:tensor(`randn; k,n; `double)        /unit normal
x:raze spiral[n]'[til k;.2*tensor z]  /generate spirals w'noise
y:til[k]where k#n                     /class
show grid[x]y

q:module seq(`sequential; (`linear; 2; 100); `relu; (`linear; 100; k))
l:loss`ce            /cross-entropy loss
o:opt(`adamw; q; .1) /Adam optimizer, weighted, learning rate .1
o:opt(`lamb; q; .01;`trustmax,1.1) 
m:model(q;l;o)

train(m;`batchsize`shuffle`metrics`tensor;(n*k;1b;`accuracy;0b))
train(m; x; y);
\ts:500 run m

test(m;`metrics;`accuracy);
-1"Accuracy on training data: ",string[testrun(m;x;y)],"%";

tensor(`randn; k,n; z)                    /new set of random variables
x:"e"$raze spiral[n]'[til k;.2*tensor z]  /rebuild spirals
-1"Accuracy using new sample: ",string[testrun(m;x;y)],"%";
