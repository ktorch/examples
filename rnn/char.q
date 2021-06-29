{key[x]set'x;}(`ktorch 2:`fns,1)[]

/ read in text, make maps from char <-> number
t:` sv read0` sv @[` vs hsym .z.f;1;:;`data],`shakespeare.txt
char:(`s#get enum:char!til count char)!char:asc distinct t

if[in[c:device[];cudadevices()]; setting`benchmark,1b] /set benchmark mode if CUDA
h:512               // size of hidden layer
s:100               // size of sequence
a:count char        // size of alphabet
t:tensor(enum t;c)  // map text -> tensor of longs, move to device
 
r:module`recur
module(r; 1; `sequential)
module(r; 2; `transform);       // define transformations for train & eval modes
module(r; 3; `sequential);      // 1st sequence used when training
module(r; 4;(`reshape; -1,s));  // reshape to matrix, 1 rows per sample in batch, cols for sequence
module(r; 3; `sequential);      // 2nd sequenctial in transform used in eval mode
module(r; 4;(`reshape; 1 1));   // make 1x1 matrix from next char
module(r; 2; (`embed; a; a));                // embedding w'characteristics of alphabet
module(r; 1; (`lstm; a; h; 2; 1b; 1b; .5));  // recurrent layer
module(r; 1; `sequential)                    // sequence for output
module(r; 2; (`drop; .5))                    // dropout
module(r; 2; (`reshape; -1,h))               // reshape to rows per sample in batch
module(r; 2; (`linear; h; a))                // linear layer from hidden to weight for each letter in alphabet
to(r;c)
m:model(r; loss`ce; opt(`adam; r))

f:{[m;t;v;i;s;w]
/x:narrow(t;0;i;w); resize(x;-1,s);  /i'th batch, reshape to batches x sequence
 x:narrow(t;0;i;w);                  /i'th batch, reshape to batches x sequence
 y:narrow(t;0;i+1;w);                /target is i'th batch shifted by one
 vector(v;0;x);                      /vector w'x & previous hidden state, if any
 zerograd m;
 use[v]forward(m;v;til tensorcount v); /save model output & hidden state
 x:tensor(v;0);                        /model output
 a:tensor z:loss(m;x;y);               /calculate loss vs actual char sequence
 backward z;
 clip(m;5);                            /clip gradients
 step m;                               /Adam optimizer step
 free'[(x;y;z)];
 a}                                    /return loss as scalar (vector v is updated w'output & hidden state)
 
msg:{-2 "Epoch: ",(3$string x)," ",(10$string"v"$.z.T)," training error: ",.Q.fmt[6;3]y;}

epoch:{[m;t;s;w;e]
 n:size[t]0; i:rand w*:s; a:j:0; v:vector 0;
 while[n>i+1+w; a+:f[m;t;v;i;s;w]; i+:w; j+:1];
 free v; msg[e]a%j; e+1}

nextchar:{[m;c;t;v;x]
/m:model, c:compute device, t:temp, v:vec w'input & hidden state, x:char
 vector(v;0;tensor(x;c));
 use[v]forward(m;v;til tensorcount v);
 multinomial first softmax vector[(v;0)]%t}
 
generate:{[m;c;t;n;x] 
 training(m;0b); g:nextchar[m;c;t]v:vector 0e;         /set eval mode, define fn g to get next char
 g'[p:(-1+count x)#x]; r:n g\last x; free v; char p,r} /initialize hidden state, generate n chars

20 epoch[m;t;s;128]/1;    /20 passes w'batch size of 128

/ with prime string "Where" & temperature of 0.8, generate 1,000 chars
-2 "\n",generate[m;c;.8;1000]enum "Where"; 
