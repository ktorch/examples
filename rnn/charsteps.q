{key[x]set'x;}(`ktorch 2:`fns,1)[]

/ read in text, make maps from char <-> number
t:` sv read0` sv @[` vs hsym .z.f;1;:;`data],`shakespeare.txt
char:(`s#get enum:char!til count char)!char:asc distinct t

if[in[c:device[];cudadevices()]; setting`benchmark,1b] /set benchmark mode if CUDA
h:512               / size of hidden layer
s:100               / size of sequence
w:128               / batch size
a:count char        / size of alphabet
t:tensor(enum t;c)  / map text -> tensor of longs, move to device
z:enum"Where"       / priming phrase for generating text
 
r:module`recur
module(r; 1; `sequential)
module(r; 2; `transform);       / define transformations for train & eval modes
module(r; 3; `sequential);      / 1st sequence used when training
module(r; 4;(`reshape; -1,s));  / reshape to matrix, 1 row per sample in batch, cols for sequence
module(r; 3; `sequential);      / 2nd sequential in transform used in eval mode
module(r; 4;(`reshape; 1 1));   / make 1x1 matrix from next char
module(r; 2; (`embed; a; a));                / embedding w'characteristics of alphabet
module(r; 1; (`lstm; a; h; 2; 1b; 1b; .5));  / recurrent layer accepts output of transformation
module(r; 1; `sequential)                    / sequence for output
module(r; 2; (`drop; .5))                    / dropout
module(r; 2; (`reshape; -1,h))               / reshape to rows per sample in batch
module(r; 2; (`linear; h; a))                / linear layer from hidden to weight for each letter in alphabet
to(r;c)
m:model(r; loss`ce; opt(`adam; r))

f:{[m;t;v;i;w]
 x:narrow(t;0;i;w);        /i'th batch
 y:narrow(t;0;i+1;w);      /target is i'th batch shifted by one
 vector(v;0;x);            /vector w'x & previous hidden state, if any
 zerograd m;
 use[v]forward(m;v);          /get model output & hidden state
 backward z:loss(m;(v;0);y);  /calculate loss vs actual char sequence
 clip(m;5);                   /clip gradients
 step m;                      /Adam optimizer step
 free y;
 return z}                    /return loss (vector v is updated w'output & hidden state)
 
nextchar:{[m;c;t;v;x]
/m:model, c:compute device, t:temp, v:vec w'input & hidden state, x:char
 vector(v;0;tensor(x;c));                    /set next char in vector
 use[v]eforward(m;v);                        /vector w'forward calc (output,hidden state)
 multinomial first softmax vector[(v;0)]%t}  /pick most likely character from model output

generate:{[m;c;t;n;x]
 g:nextchar[m;c;t]v:vector 0e;                /define fn g to get next char
 g'[p:-1_x]; r:n g\last x; free v; char p,r}  /initialize hidden state, generate n chars

gen:{[c;t;n;x;m]"   ",(last[n,where" "=x]#x:ssr[generate[m;c;t;n;x];"\n";" "]),".."}[c;.8;60;z]
msg:{-2 "Epoch:",(-3$string y),(-10$string"v"$.z.T),"  Error: ",.Q.fmt[5;3;z],gen x;}

epoch:{[m;t;w;e]
 n:size[t]0; i:rand w; a:j:0; v:vector 0e;
 while[n>i+1+w; a+:f[m;t;v;i;w]; i+:w; j+:1];
 free v; msg[m;e;a%j]; e+1}

msg[m;0;0N]                        /initial timestamp, text generation from unfitted model
20 epoch[m;t;s*w]/1;               /20 passes
-2 "\n",generate[m;c;.8;1000;z];   /using prime string & temperature 0.8, generate 1,000 chars
