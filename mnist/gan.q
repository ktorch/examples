p:first` vs hsym .z.f                      /path of this script
system"l ",1_string` sv p,`mnist.q         /load data w'mnist.q (assume same dir)
{key[x]set'get x}(`ktorch 2:`fns,1)[];     /define ktorch api fns

w:60
n:100 256 128 64;
if[in[c:device[];cudadevices()]; setting`benchmark,1b]
build:{to(x:module x;y); model(x; loss`bce; opt(`adam;x;.0002;.5))}

a:`pad`bias!(1;0b)
g:seq(`sequential;                                                  / output size:
      (`convtranspose2d;n 0;n 1;4;1_a); (`batchnorm2d;n 1); `relu;  / 256 x  4 x  4
      (`convtranspose2d;n 1;n 2;3;2;a); (`batchnorm2d;n 2); `relu;  / 128 x  7 x  7
      (`convtranspose2d;n 2;n 3;4;2;a); (`batchnorm2d;n 3); `relu;  /  64 x 14 x 14
      (`convtranspose2d;n 3;  1;4;2;a); `tanh)                      /   1 x 28 x 28
g:build[g]c

a:`bias,0b
d:seq(`sequential;                                                       / output size:
      (`conv2d;  1;n 3;4;2;1;a); (`leakyrelu; 0.2);                      /  64 x 14 x 14
      (`conv2d;n 3;n 2;4;2;1;a); (`batchnorm2d;n 2); (`leakyrelu; 0.2);  / 128 x  7 x  7
      (`conv2d;n 2;n 1;4;2;1;a); (`batchnorm2d;n 1); (`leakyrelu; 0.2);  / 256 x  3 x  3
      (`conv2d;n 1;  1;3;1;0;a); `sigmoid; (`flatten;0))                 /   1 x  1 x  1
d:build[d]c

resize(x:tensor(-1+"e"$mnist[`x]%127.5; c); -1 1 28 28) /4d tensor of images x channels x 28 x 28
b:size[x][0]div w                                       /no. of batches of size w
t:(tensor(;w;c)@)'[`empty`zeros`ones]
z:tensor(`empty;w,n[0],1 1;c)  // tensor will be repopulated w'std normal variables w'each batch
Z:tensor(`randn;w,n[0],1 1;c)  // fixed set of normal variables to show progress of generator
 
f:{[d;g;t;x;z;w;i]  /d:discriminator, g:generator, t:targets, x:images, z:noise, w:batch size, i:index
 batch(x;i;w);
 uniform(t 0;0.8;1.0); zerograd d; l0:backward(d;x;t 0); /train D w'real images, D(x) ~ 1
 normal z; dx:detach gx:forward(g;z);                    /train D w'generated images D(G(z)) ~ 0
 l1:backward(d;dx;t 1); free dx; step d;
 zerograd g; l2:backward(d;gx;t 2); free gx; step g;     /train G, aim for D(G(z)) ~ 1
 (l0+l1),l2}                                             /return D & G loss
 
msg:{-2 "Epoch: ",(4$string x)," ",(12$string"v"$.z.T),
     raze("Median loss for discriminator:";"\tgenerator:"),'.Q.fmt[6;3]'[med y];}

// generate images, no gradients, scale back to 0-255, write to png for each epoch
f1:{` sv x,`out,`$("gan","0"^-2$string y),".png"}p
f2:{[e;g;z] a:"h"$127.5*1+evaluate(g;z); png(f1 e;makegrid(a;10;6;2;255));}

do[20; msg[e+:1]f[d;g;t;x;z;w]'[til b]; f2[e;g;Z]; restore x; shuffle x]
