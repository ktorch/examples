{key[x]set'x}(`ktorch 2:`fns,1)[];  /define k api to pytorch in root namespace

/ --------------------------
/ build train/test data sets
/ --------------------------
a:2  /number of digits
x:{i:til prd 2#x:prd x#10; j:i div x; k:i mod x; (j;k;j+k)}a
x:flip raze vs'[(a+0 0 1)#'10;x]
x:{x(0,floor .9*n)_neg[n]?n:count x}x;  /90% for training, 10% for testing
-2 "sample data sequences:"; show -5?x 0; -2"";

/ ---------------------------------------------------------------------------------------------------
/ define small GPT model based on these parameters
/ v-vocabulary size, w-batch size, d-embedding dim, h-heads, n-sequence length, p-dropout probability
/ define 3 parts of the gpt model:
/ 1-embedding layer, 2-transformer block, 3-final decode & transformations
/ ---------------------------------------------------------------------------------------------------
v:10; w:500; d:128; h:4; n:3*a; p:.1

emb:{[v;d;n;p]seq(`sequential; (`embedseq;v;d;n); (`drop;p))}

/block: self attention, then linear layers sandwiching an activation fn, dropout at end
/ the 2 parts are residual layers: output is x+q2 x:x+q1 x
blk:{[d;h;p]
 q1:seq(`sequential; (`selfattention;d;h;p;1b); (`drop;p));
 q2:seq(`sequential; (`layernorm;d);(`linear;d;d*4;0b); `gelu; (`linear;d*4;d); (`drop;p));
 (`sequential; (`residual;q1); (`residual;q2))}

end:{[d;v]
 q:seq(`sequential; (`layernorm;`norm;d); (`linear;`decode;d;v;0b)); /output logits for each token
 q,enlist(`transform; seq(`sequential; (`reshape;-1,v));             /train: reshape to match target
                      seq(`sequential; (`select;1;-1)))}             /eval: take last prediction

q:`seqlist, enlist emb[v;d;n;p]
q:2 {y,enlist x}[blk[d;h]p]/q
q:q,enlist end[d;v]
q:module q

c:device[];                                   /get default device
if[c in cudadevices(); setting`benchmark,1b]  /set benchmark mode if CUDA
to(q;c)                                       /move module to cuda if avail

/ ---------------------------------------------------------------
/ init linear & embed wt to normal w'lower stddev, zero-fill bias
/ no wt decay for bias or embed/norm wts
/ ---------------------------------------------------------------
p:parmtypes q  / map of parameter name -> module type
{$[y like"*.weight"; normal(x;y;0;.02); zeros(x;y)]}[q]'[where p in`embed`linear];
p:{(b; x except b:x where (x in y)|x like"*.bias")}[key p;where p in`layernorm`embed`embedpos]
o:opt(`adamw; (); .0006; .9; .95) /empty adam optimizer w'learning rate & beta settings
opt(o;0;(q;p 0);`decay,0.0)       /no weight decay: linear bias, embed or normalizations
opt(o;1;(q;p 1);`decay,0.1)       /weight decay for remaining parameters
m:model(q; loss`ce; o);           /create model: module, optimizer and loss

/ -------------------------------------------------------------------------
/ pick     - pick next digit based on maximum weight from model output rows
/ predict  - compute next digit in batches of 1,000
/ match    - create table of predictions and actual values by dataset
/ accuracy - return fraction of accurate predictions
/ -------------------------------------------------------------------------
pick:{[m;u;x]argmax(evaluate(m;x;u);1)}
predict:{[m;u;x]x,'raze{[m;u;x;i]pick[m;u;x i]}[m;u;x]'[0N 1000#til count x]}

match:{[m;a;u;x]
 flip update ok:actual=predict from
      update actual:a+b from
      `a`b`predict!10 sv'(a*til 3)_flip(a+1) predict[m;u]/(2*a)#'x}

accuracy:{[m;a;u;x]exec sum[ok]%count ok from match[m;a;u;x]}

/ ------------------------------------------------------------------------------
/ msg     - output message of epoch, iteration, learning rate, loss, etc.
/ iter    - handle a single iteration within a epoch
/ epoch   - process the entire dataset in batches
/ ------------------------------------------------------------------------------
msg:{
 s:"epoch: ",-3$string x`e;
 s,:"  iter: ",(-4$string x`i)," of ",string x`it;
 s,:"  lr: ",.Q.fmt[7;5]x`lr;
 s,:"  loss:",.Q.fmt[6;3]x`l;
 s,:"  time: ",string "t"$j:.z.P-x`t;
 s,:"  ",.Q.fmt[5;1;x[`i]%1e-9*j]," iter/sec";
 2 s,$[</[x`i`it]; "\r"; ""];}

iter:{[m;a;x;u;s;i]                /process batch:
 y:1_'x@:i; x:-1_'x;               /w sequences of n+1 length
 y[;til -1+a*2]:-100;              /-100 to mask loss before output part of sequence
 nograd m;                         /set gradients to none
 s[`l]:backward(m; (x;u); raze y); /calculate model output,loss & gradients
 s[`i]+:1;                         /update iterations
 clip(m;1);                        /clip gradients to norm of 1
 msg s; step m; s}                 /display state, update parms & return state

epoch:{[m;a;x;u;w;s]
 i:(0N,w)#neg[i]?i:count x 0;    /groups of w indices (for training data, shuffled)
 s[`e`i`t]:(1+s`e; 0; .z.P);     /update current epoch, reset counter & start time
 s:iter[m;a;x 0;u]/[s;i];        /process all batches and return state
 -2 .Q.fmt[8;1;100*accuracy[m;a;u;x 1]],"%";
 s}

/ ---------------------------------------------------------------------
/ define training state: epochs, iters per epoch, initial learning rate
/ ---------------------------------------------------------------------
s:`ep`it`lr ! (50; ceiling(count[x 0]-1+n)%w; first lr m)
s,:`e`t`i`l ! 4#0    /current epoch, start time, iter & loss

u:tensor(triu((2#n)#-0we;1);c)   /upper triangular attention mask
s:s.ep epoch[m;a;x;u;w]/s        /train for given epochs

t:raze{`dataset xcols update dataset:x from y}'[`train`test;match[m;a;u]'[x]]
e:reverse exec sum not ok by dataset from t;
-2 {"\nmismatches in ",", "sv": "sv/:flip string(key x;get x)}e;
{if[count x;show x]}select from t where not ok
