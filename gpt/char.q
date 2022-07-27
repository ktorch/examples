{key[x]set'x}(`ktorch 2:`fns,1)[];   /define k api to pytorch in root namespace

/ ---------------------------------------------
/ read in text, make maps from char <-> number
/ ---------------------------------------------
t:` sv read0` sv @[` vs hsym .z.f;1;:;`data],`shakespeare.txt
char:(`s#get enum:char!til count char)!char:asc distinct t
t:enum t

/ ---------------------------------------------------------------------------------------------------
/ define small GPT model based on these parameters
/ v-vocabulary size, w-batch size, d-embedding dim, h-heads, n-sequence length, p-dropout probability
/ define 3 parts of the gpt model:
/ 1-embedding layer, 2-transformer block, 3-final decode & transformations
/ ---------------------------------------------------------------------------------------------------
v:count char; w:200; d:512; h:8; n:128; p:.1

emb:{[v;d;n;p]seq(`sequential; (`embedseq;v;d;n); (`drop;p))}

/ block: self attention, then linear layers sandwiching an activation fn, dropout at end
/ the 2 parts are residual layers: output is x+q2 x:x+q1 x
blk:{[d;h;p]
 q1:seq(`sequential; (`selfattention;d;h;p;1b); (`drop;p));
 q2:seq(`sequential; (`layernorm;d);(`linear;d;d*4;0b); `gelu; (`linear;d*4;d); (`drop;p));
 (`sequential; (`residual;q1); (`residual;q2))}

end:{[d;v]
 q:seq(`sequential; (`layernorm;`norm;d); (`linear;`decode;d;v;0b)); /output logits for each token
 q,enlist(`transform; seq(`sequential; (`reshape;-1,v));             /train: reshape to match target
                      seq(`sequential; (`select;1;-1)))}             /eval: take last prediction

q:`seqlist, enlist emb[v;d;n;p]  / token & positional embedding
q:8 {y,enlist x}[blk[d;h]p]/q    / 8 transformer blocks
q:q,enlist end[d;v]              / end decoder and transforms
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
o:opt(`adamw; (); .0006; .9; .95) /empty adam optimizer w'initial learning rate & beta settings
opt(o;0;(q;p 0);`decay,0.0)       /no weight decay: linear bias, embed or normalizations
opt(o;1;(q;p 1);`decay,0.1)       /weight decay for remaining parameters
m:model(q; loss`ce; o);           /create model: module, optimizer and loss

/ ------------------------------------------------------------------------------
/ lrdecay - cosine learning rate decay, limit to 10% of inital rate
/ msg     - output message of epoch, iteration, learning rate, loss, etc.
/ iter    - handle a single iteration within a epoch
/ epoch   - process the entire dataset in batches
/ ------------------------------------------------------------------------------
lrdecay:{x[`lr] * .1|.5*1+cos %/[@[x[`it]*-1 0+x`e`ep; 0; +; x`i]]*acos -1} 

msg:{
 s:"epoch: ",-2$string x`e;
 s,:"  iter: ",(-4$string x`i)," of ",string x`it;
 s,:"  lr: ",.Q.fmt[7;5]x`r;
 s,:"  loss:",.Q.fmt[6;3]x`l;
 s,:"  time: ",string "v"$j:.z.P-x`t;
 s,:"  ",.Q.fmt[4;1;x[`i]%1e-9*j]," iter/sec";
 2 s,$[</[x`i`it]; "\r"; "\n"];}

iter:{[m;t;u;n;s;i]                        /process batch:
 x:t i+\:til n+1;                          /set w sequences of n+1 length;
 nograd m;                                 /set gradients to undefined tensor
 s[`l]:backward(m; (-1_'x;u); raze 1_'x);  /calculate model output,loss & gradients
 s[`i]+:1;                                 /update iterations
 s[`r]:lrdecay s;                          /decay learning rate based on progress
 clip(m;1);                                /clip gradients to norm of 1
 msg s; lr(m;s`r); step m; s}              /display state, set learning rate, update parms & return state

epoch:{[m;t;u;n;w;s]
 i:(0N,w)#neg[i]?i:count[t]-1+n;  /rows: w starting indices of sequences length n+1
 s[`e`i`t]:(1+s`e; 0; .z.P);      /update current epoch, reset counter & start time
 iter[m;t;u;n]/[s;i]}             /process all batches and return state
 
/ ------------------------------------------------------------------------------
/ pick - scale wts, pick top k if k non-zero, sample or take max
/ generate - given pick function f, model, max sequence, mask, length and phrase
/            generate characters from the model
/ ------------------------------------------------------------------------------
pick:{[t;k;s;x] /generate next char given temp, top k, sample flag & logits
 if[not t=1; x%:t];                                       /scale by temperature
 if[k; x:@[count[x]#max 0#x; j 1; : ;first j:topk(x;k)]]; /set -inf outside top k
 x:softmax x;                                             /output -> probabability
 $[s; multinomial x; argmax x]}                           /sample or largest prob

generate:{[f;m;n;u;w;x]
 x:enlist enum x; / starting phrase to 1-row batch of ints
 g:{[f;m;n;u;x]x,'f'[evaluate(m;(neg n&count'[x])#'x;u)]};
 y:char first w g[f;m;n;u]/x}

/ ------------------------------------------------------------------------------
/ define training state: epochs, iters per epoch, initial learning rate
/ ------------------------------------------------------------------------------
s:`ep`it`lr ! (2; ceiling(count[t]-1+n)%w; first lr m)
s,:`e`t`r`i`l ! 5#0    /current epoch, start time, learning rate, iter & loss

u:tensor(triu((2#n)#-0we;1);c) /upper triangular attention mask
s:s.ep epoch[m;t;u;n;w]/s      /train for given epochs

f:pick[1;10;1b]         /pick next char with temperature=1, top 10, sample
g:generate[f;m;n;u]     /generate given pick fn, model, max sequence & mask
-2 "\n",g[500]"O God, O God!";  / generate 500 chars
