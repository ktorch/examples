{key[x]set'x;}(`ktorch 2:`fns,1)[];                     /define pytorch interface
d:first` vs hsym .z.f                                   /dir of this script
b:3*f:5                                                 /backcast 15,forecast 5
x:last flip("MJ";1#",")0:` sv d,`data`milk.csv          /assume data/ with .csv in same dir as script
x:(0,floor .8*count x)_x@:til[b+f]+/:til 1+count[x]-b+f /create train & test data (80/20% split)
x%:a:max raze first x                                   /divide by max of train set
x:flip''[(0,b)_/:flip'["e"$x]]                          /backcast & forecast for train & test sets

block:{[u;n;b;f;t] /u:hidden units, n:layers, b:backcasts, f:forecasts, t:thetas
 r:(`sequential; (`linear;b;u); `relu);
 r,:(2*n-1)#((`linear;u;u); `relu);
 seq[r],generic[u;b;f;t]}

generic:{[u;b;f;t] /u:hidden units, b:backcasts, f:forecasts,t:thetas
 a:`relu; m:`linear,'((`theta;u;t;0b); (`backcast;t;b); (`forecast;t;f));
 enlist(`fork;seq(`sequential`bc;m 0;a;m 1);seq(`sequential`fc;m 0;a;m 2))}

if[in[c:device[];cudadevices()]; setting`benchmark,1b]  /set benchmark mode if CUDA available
m:module`nbeats,raze block[256;4;b;f]''[3#'4 8]         /2 stacks of 3 blocks each, 4 layers per block
to(m;c)
m:model(m;loss`mse;opt(`adamw;m;`decay,.1))

v:x{to(v:vector x;y);v}'c                               /two x,y vectors(train & test), on gpu if avail
epoch:{[m;v;e;w]raze e{y:train(x)}[(m;v 0;w;0b)]\()}    /e epochs w'batch size w, return loss at each gradient step
msg:{string["v"$.z.T],raze(" epochs: ";"\tgradient steps: ";"\tloss: ";"\ttest loss: "),'-4 -5 7 7$string x,1e-5*"j"$1e5*y}
run:{[m;v;e;w;n] l:epoch[m;v;e;w]; -2 msg[n+:e,count l]avg[l],evaluate(m;v 1;`loss); lr(m;.95*lr m); n}
\ts 25 run[m;v;100;30]/0   / 25 runs of 100 epoch's each, batch size of 30

y:a*vector(v 1;1);
yhat:a*evaluate(m;v 1)

pct:{.1*"j"$1000*(min;max;avg;med)@\:raze -1+y%x}
-2 raze("\nprediction errors, lo: ";", hi: ";", mean: ";", median: "),'string[pct[y]yhat],'"%"; -2"";

t:`period`y`yhat!/:raze{flip(x;y;z)}'[til count y;y;yhat]
t:update pct:100*diff%y from update diff:yhat-y from t
rnd:(.1*"j"$10*)

-2 "\nhighest absolute errors:"; show rnd select from t where {x=max x}(avg;abs pct)fby period
-2 "\nlowest absolute errors:";  show rnd select from t where {x=min x}(avg;abs pct)fby period
-2 "\nfinal period:";            show rnd select from t where period=max period

\
 =ensemble with backcast of 2-7x forecast
 -seasonal & trend blocks
 -share of weights in stack (across blocks)
 -last obs & shuffle
