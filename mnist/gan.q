p:first` vs hsym .z.f                        /path of this script
system"l ",1_string` sv p,`mnist.q           /fns for reading MNIST files
{key[x]set'get x}(`ktorch 2:`fns,1)[];       /define ktorch api fns

c:device[]                                   /CUDA compute device if available
if[c in cudadevices(); setting`benchmark,1b] /set benchmark mode if CUDA
x:tensor(-1+"e"$mnist[][`x]%127.5; c)        /rescale images, move to compute device
resize(x; -1 1 28 28)                        /4d tensor(images x channels x 28 x 28)
b:size[x][0]div w:60                         /no. of batches of size w
n:100 256 128 64                             /convolution sizes

gan:{to(x:module seq `sequential,x;y); model(x; loss`bce; opt(`adam;x;.0002;.5))}

/ build generator
a:`pad`bias!(1;0b)
g :((`convtranspose2d;n 0;n 1;4;1_a); (`batchnorm2d;n 1); `relu)  / 256 x  4 x  4
g,:((`convtranspose2d;n 1;n 2;3;2;a); (`batchnorm2d;n 2); `relu)  / 128 x  7 x  7
g,:((`convtranspose2d;n 2;n 3;4;2;a); (`batchnorm2d;n 3); `relu)  /  64 x 14 x 14
g,:((`convtranspose2d;n 3;  1;4;2;a); `tanh)                      /   1 x 28 x 28
g:gan[g]c

/ build discriminator
a:`bias,0b
d :((`conv2d;  1;n 3;4;2;1;a); (`leakyrelu; 0.2))                      /  64 x 14 x 14
d,:((`conv2d;n 3;n 2;4;2;1;a); (`batchnorm2d;n 2); (`leakyrelu; 0.2))  / 128 x  7 x  7
d,:((`conv2d;n 2;n 1;4;2;1;a); (`batchnorm2d;n 1); (`leakyrelu; 0.2))  / 256 x  3 x  3
d,:((`conv2d;n 1;  1;3;1;0;a); `sigmoid; (`flatten;0))                 /   1 x  1 x  1
d:gan[d]c

t:(tensor(;w;c)@)'[`empty`zeros`ones]
z:tensor(`empty;w,n[0],1 1;c)  / tensor will hold std normal values w'each batch
Z:tensor(`randn;w,n[0],1 1;c)  / fixed set of random values to show generator progress
test(g; `batchsize`metrics; w,`output); test(g; Z);
 
/ -------------------------------------------------------------------
/ fit1: train discriminator on mnist to recognize as real images
/ fit2: train discriminator to recognize generated images as not real
/ fit3: train generator by w'discriminator accepting generated images 
/ fit:  handle training of discriminator and generator
/ -------------------------------------------------------------------
fit1:{[d;x;y] nograd d; uniform(y;.8;1); backward(d;x;y)}
fit2:{[d;x;y] x:detach x; l:backward(d;x;y); free x; step d; l}
fit3:{[d;g;x;y] nograd g; l:backward(d;x;y); free x; step g; l}

fit:{[d;g;t;x;z;w;i]
/d:discriminator, g:generator, t:targets, x:images, z:noise, w:batch size, i:index
 batch(x;w;i);             /take i'th subset of MNIST images
 l1:fit1[d;x;t 0];         /train discriminator w'real images
 normal z; x:forward(g;z); /generate images from noise
 l2:fit2[d;x;t 1];         /train d w'generated images
 l3:fit3[d;g;x;t 2];       /train generator w'discriminator accepting generated images
 (l1+l2),l3}               /return discriminator & generator loss

/ ---------------------------------------------------------------------
/ msg: print losses for each epoch
/  f1: file name for test grid of generated digits, e.g. out/cgan01.png
/  f2: generate test grid and write to .png file
/ gif: attempt to create .gif file of each epoch's test digits
/ ---------------------------------------------------------------------
msg:{-2 "Epoch: ",(4$string x)," ",(12$string"v"$.z.T),
     raze("Median loss for discriminator:";"\tgenerator:"),'.Q.fmt[6;3]'[med y];}

/ generate images, no gradients, scale back to 0-255, write to png for each epoch
f1:{` sv x,`out,`$("gan","0"^-2$string y),".png"}p
f2:{[e;g] a:"h"$127.5*1+testrun g; png(f1 e;makegrid(a;10;6;2;255));}

gif:{  /use convert utility if found to build .gif of epoch samples
 if[not count c:first @[system;"which convert";""]; :()];
 c:" " sv(c; "-delay 100 -loop 0"; " " sv 1_'string f1'[1+til y]; 1_string` sv x,`out,`gan.gif); 
 @[system; c; ""];}

/ --------------------------------------------------------
/ run GAN for n epochs, build gif of snapshots
/ --------------------------------------------------------
n:20
-2 raze("Epochs: "; ", batch size: ";", iterations per epoch: "),'string(n;w;b);
do[n; msg[e+:1]fit[d;g;t;x;z;w]'[til b]; f2[e;g]; shuffle x]; gif[p]20

f:@[string(` sv p,`out`),(last` vs)'[f1'[1,n]],`gan.gif;0;1_]
-2"\nGenerated digits in dir ",f[0],", ",f[1]," - ",f[2],", ",f 3;
