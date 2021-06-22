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
module(r; 2; (`embed; a; a));
module(r; 1; (`lstm; a; h; 2; 1b; 1b; .5));
module(r; 1; `sequential)
module(r; 2; (`drop; .5))
module(r; 2; (`reshape; -1,h))
module(r; 2; (`linear; h; a))
to(r;c)
m:model(r; loss`ce; opt(`adam; r))

f:{[m;t;v;i;s;w]
 x:narrow(t;0;i;w); resize(x;-1,s);  /i'th batch, reshape to batches x sequence
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
 a}
 
msg:{-2 "Epoch: ",(3$string x)," ",(10$string"v"$.z.T)," training error: ",.Q.fmt[6;3]y;}

epoch:{[m;t;s;w;e]
 n:size[t]0; i:rand w*:s; a:j:0; v:vector 0;
 while[n>i+1+w; a+:f[m;t;v;i;s;w]; i+:w; j+:1];
 free v; msg[e]a%j; e+1}

nextchar:{[m;c;v;t;x]
 vector(v;0;tensor(1 1#x;c));
 use[v]forward(m;v;til tensorcount v);
 multinomial first softmax vector[(v;0)]%t}
 
predict:{[m;c;t;n;x] training(m;0b); v:vector 0e; r:char n nextchar[m;c;v;t]\x; free v; r}

20 epoch[m;t;s;128]/1;    /20 passes w'batch size of 128

-2 "\n",predict[m;c;.8;1000]enum"A";
