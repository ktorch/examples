/ given path w'MNIST files, create dictionary of train/test images & labels
/ download from http://yann.lecun.com/exdb/mnist/
mnist:hsym`$getenv[`HOME],"/data/mnist"
mnist:mnist{` sv x,`$"-"sv string y}'cross/[(`train`t10k;`images`labels,'`idx3`idx1;`ubyte)]
mnist:`x`y`X`Y!{$[x like"*images*";16_;8_]"h"$read1 x}'[mnist]

\
/check a random digit from training data
{show y i:rand count y; -2"* "0=28 28#x til[784]+784*i;} . mnist`x`y
