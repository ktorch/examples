{key[x]set'x}(`ktorch 2:`fns,1)[];   /load PyTorch interface into root dir
a:first` vs hsym .z.f                /path of this script
system"l ",1_string` sv a,`mnist.q   /fns for reading MNIST files
d:mnist` sv a,`data
@[`d;`x`X;{(div[count x;prd s],s:1 28 28)#x}];
d[`mean`std]:meanstd(d`x;0 2 3)

i:1              /input number of channels
s:28             /size of the images
k:10             /number of classes
p:.1             /probablity used with dropout layers

/cct 2/3x2 small model, around 280K parameters, 2 encoder layers, 2 convolutions for tokenizing
t:2              /number of convolutions for tokenizer
e:128            /embedding dimension
h:2              /heads for self-attention
m:1              /multiplier of embed dim for linear layer in encoder
n:2              /number of encoder layers

/ ------------------------------------------------------------------------------
/ inp  - input transforms
/ pos  - learned positional embedding, rows (sequence length) based on 
/        size(s) of conv layer -> maxpool, e.g. 32 -> 16 or 32 -> 16 -> 8
/ conv - build convolutional layer, relu activation & max pooling
/ tok  - 1 or 2 convolutional blocks to tokenize image
/ blk  - build an encoder block given embedding dim,heads,multiplier,probs
/ enc  - build transformer encoder w'increasing droppath probalities
/ dec  - decode w'normalization, "attention pool" and final linear layer
/ init - initialize linear layers w'normal, std of .02 kaiming normal for conv
/ ------------------------------------------------------------------------------
inp:{
 r:((`randomcrop;`crop;x;y);(`zscore;`zscore;z 0;z 1));
 enlist[`transform`input],{seq`sequential,x}'[2 -1#\:r]}

pos:{[e;t;s](`residual`position; (`sequential; enlist(`embedpos;`emb;n*n:s div 2*t;e)))}

conv:{((`conv2d;x 0;y;z;3;1;1;`bias,0b); `relu,x 1; (`maxpool2d;x 2;3;2;1))}
tok:{
 s:("conv";"maxpool";"relu"); s:`$$[x=1;s;s,\:/:"12"];
 $[x=1; conv[s;y;z]; conv[s 0;y;64],conv[s 1;64;z]],
  ((`flatten;`flat;2;3); `transpose`transpose)}

/CCT block w'normalization out of order, github.com/SHI-Labs/Compact-Transformers/issues/25
blk1:{[e;h;m;p1;p2]
 a:seq(`sequential; (`selfattention;`attn;e;h;p1;1b); (`droppath;`drop;p2));
 b:seq(`sequential; (`linear;`linear1;e;e*m); `gelu`gelu; (`linear;`linear2;e*m;e); (`droppath;`drop;p2));
 (`sequential; (`residual`resid1;a); enlist(`layernorm;`norm;e); (`residual`resid2;b))}

blk:{[e;h;m;p1;p2]
 a:seq(`sequential; (`selfattention;`attn;e;h;p1;1b); (`droppath;`drop;p2));
 b:seq(`sequential; (`layernorm;`norm;e); (`linear;`linear1;e;e*m); `gelu`gelu; (`linear;`linear2;e*m;e); (`droppath;`drop;p2));
 (`sequential; (`residual`resid1;a); (`residual`resid2;b))}

enc:{[e;h;m;n;p]enlist[`seqlist`blocks],blk[e;h;m;p]'[(n-1) ((p%n-1)+)\0.0]}

dec:{[e;k]
 (`seqnest`end;
   enlist(`layernorm;`norm;e);
  (`seqjoin`join; seq(`sequential; (`linear;`attnpool;e;1); (`softmax;`softmax;1); `transpose`transpose); `matmul`mul);
   enlist(`squeeze;`squeeze;-2);
   enlist(`linear;`fc;e;k))}

init:{[q]
 p:where each parmtypes[q]=/:`linear`conv2d;
 {$[y like"*.weight"; normal(x;y;0;.02); zeros(x;y)]}[q]'[p 0];
 {if[y like"*.weight"; knormal(x;y)]}[q]'[p 1];}

q:`inp`tok`pos`enc`dec!()
q.inp: inp[s;3;d`mean`std]                          /input transform(s)
q.tok: seq enlist[`seqnest`token],tok[t;i;e]        /tokens from convolution(s)
q.pos: pos[e;t;s]                                   /position embedding
q.enc: enc[e;h;m;n;p]                               /encoder blocks
q.dec: dec[e;k]                                     /decoder layers
q:{module(x;1;module y); x}/[module`sequential;q]

if[in[c:device[];cudadevices()]; setting`benchmark,1b] /set benchmark mode if CUDA
init q
to(q; c)

g:parmtypes q
g:{(b;x except b:x where(x in y)|x like"*.bias")}[key g;where g in`embedpos`layernorm]
o:opt(`adamw;();.0005;.9;.95)   /empty parameters used to initialize optimizer
opt(o;0;(q;g 0);`decay,0.0)    /no decay for embedding, normalization & biases
opt(o;1;(q;g 1);`decay,0.1)    /small decay for other layers
m:model(q; loss(`ce;`smoothing,.2); o)

w:50 1000
train(m; `batchsize`shuffle; (w 0;1b)); n:train(m; d`x; "j"$d`y);
test(m; `batchsize`metrics; (w 1; `loss`accuracy)); n:n,test(m; d`X; "j"$d`Y)
-2 ", "sv("train: ";"test: "),'{raze string[(x;y)],'(" batches of ";" images")}'[n;w];

r:{(1+til y)!x*.5*1+cos acos[-1]*til[n]%n:y}[.0005]40  /cosine decay of learning rate
fmt:.Q.fmt'[4 9 9 7 6;0 6 6 4 2] /format epoch, rate, loss, test loss & accuracy
msg:{-1 raze[("";"  lr:";"  loss:";"  test:";"  accuracy:"),'fmt[x]],"%   ",string"v"$.z.T;}
fit:{[m;r;i] lr(m;r@:i); msg i,r,raze`run`testrun@\:m; i+1}

\ts count[r] fit[m;r]/1;

/build table of mismatches in test dataset, convert to .png with labels
test(m;`metrics;`predict)
s:digits[] / define a set of standardized digits for labeling
t:asc{([]y:x;yhat:y;ind:til count x)where not x=y}[d`Y]testrun m
y:exec yhat by y from t            /mismatches by digit
x:exec ind by y from t             /indices in test data
n:{max[x]-x}count'[x]              /padding count to match max width (typically for '9')
x:first''[d[`X]get x]              /test images where prediction wrong(by digit)
g:z,/:s[y],'n#\:z:1 28 28#0h       /blank leading column, blanks to pad to same width
g:g,'s[0N 1#key y],'x,'n#\:z       /join labels w'image list padded with blank images
g:makegrid(raze g; 2*count y; 1+max count'[y]; 2; 255)  /re-arrange into single grid of images

-2 "\nmismatches:"; show y
-2 "\ngrid of mismatches: ",1_string png(` sv a,`out`cct.png;g);
