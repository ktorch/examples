cifar10:{
 /if no data/ dir given, assume in same location as invoking script or current dir
 if[null x;  x:$[null .z.f;`:./data;` sv @[` vs hsym .z.f;1;:;`data]]];
 f:key d:` sv hsym[x],`$"cifar-10-batches-bin";
 f:` sv''d,/:'(f where f like)each("data_batch_[1-5].bin";"test_batch.bin";"batches.meta.txt");
 if[not 5 1 1~count each f; '"unable to find all CIFAR10 files in ",1_string d];
 x:(0N,1+prd s:3 32 32){x#raze read1'[y]}/:2#f;  /read in train & test data
 x:`x`y`X`Y!raze s{(x#/:1_'y;y[;0])}/:"h"$x;     / x,y for training data, X,Y for test data
 @[x;`s;:;{`$x where not x like""}read0 f . 2 0]}
