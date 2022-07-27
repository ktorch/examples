p:first` vs hsym .z.f                      /path of this script
system"l ",1_string` sv p,`mnist.q         /fns to read MNIST files
{key[x]set'get x}(`ktorch 2:`fns,1)[];     /define ktorch api fns

e:10                 /dimension of learned embedding for generator
zn:100               /number of random variables for generator
n:(e+zn),256 128 64  /size of convolutional layers

c:device[]           /get default cuda device if avail, else cpu
if[c in cudadevices(); setting(`benchmark;1b)]

/ -----------------------------------------------------------------
/ generator: join random vars w'learned embedding of matching digit
/ -----------------------------------------------------------------
a:`pad`bias!(1;0b)

g:((0; `sequential);
   (1; `seqjoin);
   (2; `sequential);          / 1st fork: random vars
   (2; `sequential);          / 2nd fork: digit
   (3; (`embed;10;e));
   (3; (`reshape;-1,e,1,1));
   (2; (`cat;1));             / join inputs
   (1; (`convtranspose2d;n 0;n 1;4;1_a));
   (1; (`batchnorm2d;n 1));
   (1; `relu);
   (1; (`convtranspose2d;n 1;n 2;3;2;a));
   (1; (`batchnorm2d;n 2));
   (1; `relu);
   (1; (`convtranspose2d;n 2;n 3;4;2;a));
   (1; (`batchnorm2d;n 3));
   (1; `relu);
   (1; (`convtranspose2d;n 3;1;4;2;a));
   (1; `tanh))

/ ----------------------------------------------------------------
/ discriminator: join images w'learned embedding of matching digit
/ ----------------------------------------------------------------
e:50         /reset embedding dimension for discriminator target
a:`bias,0b   /option(s) for convolutions

d:((0; `sequential);
   (1; `seqjoin);
   (2; `sequential);     / 1st fork: images passed through empty sequential
   (2; `sequential);     / 2nd fork: digit -> embedding -> liner -> 28 x 28
   (3; (`embed;10;e));
   (3; (`linear;e;28*28));
   (3; (`reshape;-1 1 28 28));
   (2; (`cat;1));        / join inputs
   (1; (`conv2d;2;n 3;4;2;1;a));
   (1; (`leakyrelu; 0.2));
   (1; (`conv2d;n 3;n 2;4;2;1;a));
   (1; (`batchnorm2d;n 2));
   (1; (`leakyrelu; 0.2));
   (1; (`conv2d;n 2;n 1;4;2;1;a));
   (1; (`batchnorm2d;n 1));
   (1; (`leakyrelu; 0.2));
   (1; (`conv2d;n 1;1;3;1;0;a));
   (1; `sigmoid);
   (1; (`flatten;0)))

/ ------------------------------------------------------------
/ get computing device, build generator & discriminator models
/ ------------------------------------------------------------
c:device[]           /get default cuda device if avail, else cpu
if[c in cudadevices(); setting(`benchmark;1b)]

gan:{q:module x; to(q;y); model(q; loss`bce; opt(`adam;q;.0002;.5))}
g:gan[g]c
d:gan[d]c

/ -------------------------------------------------------------------------------
/ build vectors for training & test images & labels, target tensor & random noise
/ -------------------------------------------------------------------------------
v:mnist[]`x`y                             /mnist images & targets
v[0]:resize(-1+v[0]%127.5;-1 1 28 28)     /rescale,resize: n x channel x ht x wd
b:count[v 0]div w:60                      /number of whole batches given batch size w
v:vector"ej"$v; to(v;c);                  /create vector of tensors on compute device
t:(tensor(;w;c)@)'[`empty`zeros`ones]     /targets for training 
normal z:tensor(`empty;w,zn,1 1;c)        /tensor for random variables
V:((10#tensor z)where 10#10; 100#til 10)  /inputs & targets for 10x10 test grid
V:vector V; to(V;c)                       /create test vector on compute device

/ -------------------------------------------------------------------
/ fit1: train discriminator on mnist to recognize as real images
/ fit2: train discriminator to recognize generated images as not real
/ fit3: train generator by w'discriminator accepting generated images
/ fit:  handle training of discriminator and generator
/ -------------------------------------------------------------------
fit1:{[d;t;v] nograd d; uniform(t;0.8;1.0); backward(d;v;t)}
fit2:{[d;t;x;y] x:detach x; l:backward(d;(x;y);t); free x; step d; l}
fit3:{[d;g;t;x;y] nograd g; l:backward(d;(x;y);t); free(x;y); step g; l}

fit:{[d;g;t;v;z;w;i]
 batch(v;w;i);                              /i'th subset of MNIST images & labels
 l1:fit1[d;t 0;v];                          /train on real images
 normal z; y:tensor(`randint;10;w;c,`long); /random inputs & targets
 x:forward(g;z;y);                          /generated images
 l2:fit2[d;t 1;x;y];                        /train discriminator w'generated images
 l3:fit3[d;g;t 2;x;y];                      /train generator: get discriminator to recognize as real
 (l1+l2),l3}                                /discriminator & generator loss
 
/ ---------------------------------------------------------------------
/ msg: print losses for each epoch
/  f1: file name for test grid of generated digits, e.g. out/cgan01.png
/  f2: generate test grid and write to .png file
/ gif: attempt to create .gif file of each epoch's test digits
/ ---------------------------------------------------------------------
msg:{-2 "Epoch: ",(4$string x)," ",(12$string"v"$.z.T),
     raze("Median loss for discriminator:";"\tgenerator:"),'.Q.fmt[6;3]'[med y];}

f1:{` sv x,`out,`$("cgan","0"^-2$string y),".png"}p
f2:{[g;v;i] a:"h"$127.5*1+evaluate(g;v); png(f1 i;makegrid(a;10;10;2;255));}[g]V

gif:{  /use convert utility if found to build .gif of epoch samples
 if[not count c:first @[system;"which convert";""]; :()];
 c:" " sv(c; "-delay 100 -loop 0"; " " sv 1_'string f1'[1+til y]; 1_string` sv x,`out,`cgan.gif);
 @[system; c; ""];}

/ --------------------------------------------------------
/ run conditional GAN for n epochs, build gif of snapshots
/ --------------------------------------------------------
n:20
-2 raze("Epochs: "; ", batch size: ";", iterations per epoch: "),'string(n;w;b);
do[n; msg[i+:1]fit[d;g;t;v;z;w]'[til b]; f2 i; shuffle v]; gif[p]20

f:@[string(` sv p,`out`),(last` vs)'[f1'[1,n]],`cgan.gif;0;1_]
-2"\nGenerated digits in dir ",f[0],", ",f[1]," - ",f[2],", ",f 3;
