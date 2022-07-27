widenorm:{((`batchnorm2d;x); (`relu;1b))}

wideconv:{[i;o;s;p]
 ((`conv2d;`conv1;i;o;3;s;1;`bias,0b);
  (`batchnorm2d;`bn2;o);
  (`relu;`relu2;1b);
  (`drop;`dropout;p);
  (`conv2d;`conv2;o;o;3;1;1;`bias,0b))}

wide1:{[i;o;s;p]
 q:seq`seqnest,widenorm i;
 c:seq(`sequential,wideconv[i;o;s;p]);
 d:seq((`sequential;`downsample); (`conv2d;i;o;1;s;`bias,0b));
 q,enlist(`residual; c; d)}

wide2:{[i;o;s;p]
 (`seqnest; (`residual;seq(`sequential,widenorm[o],wideconv[o;o;1;p])))}
 
wideblock:{[i;o;s;p;d] (`wide1`wide2 .\:(i;o;s;p))where 1,d}

widenet:{[d;w;p;c;z]
 d:(d-4)div 6; n:16,1 2 4*16*w;
 q:(`sequential`widenet; z; enlist(`conv2d;3;n 0;3;1;1;`bias,0b));
 q,:`seqnest,wideblock[n 0;n 1;1;p;d];
 q,:`seqnest,wideblock[n 1;n 2;2;p;d];
 q,:`seqnest,wideblock[n 2;n 3;2;p;d];
 q,enlist'[((`batchnorm2d;`batchnorm;n 3);
            (`relu;`relu;1b);
            (`avgpool2d;`avgpool;8);
            (`flatten;`flatten;1);
            (`linear;`classify;n 3;c))]}
