
.. _time:

Time series
===========

The `nbeats.q <https://github.com/ktorch/examples/blob/master/time/nbeats.q>`_ script is an implementation of the basic `N-BEATS algorithm <https://arxiv.org/abs/1905.10437>`_, a neural-network based model for univariate timeseries forecasting.

Sample data is a `small, monthly milk production dataset <https://github.com/plotly/datasets/blob/master/monthly-milk-production-pounds.csv>`_.

::

   > q examples/time/nbeats.q
   KDB+ 4.0 2020.05.04 Copyright (C) 1993-2020 Kx Systems
   l64/ 12(16)core 64037MB 

   10:38:16 epochs:  100	gradient steps:   400	loss: 0.02393	test loss: 0.00113
   10:38:19 epochs:  200	gradient steps:   800	loss: 0.00071	test loss: 0.00082
   10:38:21 epochs:  300	gradient steps:  1200	loss: 0.00052	test loss: 0.00057
   10:38:24 epochs:  400	gradient steps:  1600	loss: 0.00045	test loss: 0.00044
   10:38:26 epochs:  500	gradient steps:  2000	loss: 0.00043	test loss: 0.00039
   10:38:29 epochs:  600	gradient steps:  2400	loss: 0.00037	test loss: 0.00045
   10:38:31 epochs:  700	gradient steps:  2800	loss: 0.00036	test loss: 0.00036
   10:38:34 epochs:  800	gradient steps:  3200	loss: 0.00035	test loss: 0.00032
   10:38:36 epochs:  900	gradient steps:  3600	loss: 0.00029	test loss: 0.00043
   10:38:39 epochs: 1000	gradient steps:  4000	loss: 0.0003 	test loss: 0.00041
   10:38:41 epochs: 1100	gradient steps:  4400	loss: 0.00028	test loss: 0.00035
   10:38:44 epochs: 1200	gradient steps:  4800	loss: 0.00023	test loss: 0.00048
   10:38:46 epochs: 1300	gradient steps:  5200	loss: 0.00024	test loss: 0.00029
   10:38:49 epochs: 1400	gradient steps:  5600	loss: 0.00019	test loss: 0.00031
   10:38:51 epochs: 1500	gradient steps:  6000	loss: 0.0002 	test loss: 0.00027
   10:38:54 epochs: 1600	gradient steps:  6400	loss: 0.00016	test loss: 0.00028
   10:38:56 epochs: 1700	gradient steps:  6800	loss: 0.00017	test loss: 0.00028
   10:38:58 epochs: 1800	gradient steps:  7200	loss: 0.00015	test loss: 0.00038
   10:39:01 epochs: 1900	gradient steps:  7600	loss: 0.00014	test loss: 0.00025
   10:39:03 epochs: 2000	gradient steps:  8000	loss: 0.00013	test loss: 0.00025
   10:39:06 epochs: 2100	gradient steps:  8400	loss: 0.00012	test loss: 0.00023
   10:39:08 epochs: 2200	gradient steps:  8800	loss: 0.00011	test loss: 0.00024
   10:39:11 epochs: 2300	gradient steps:  9200	loss: 0.00011	test loss: 0.00028
   10:39:13 epochs: 2400	gradient steps:  9600	loss: 0.0001 	test loss: 0.00023
   10:39:16 epochs: 2500	gradient steps: 10000	loss: 9e-05  	test loss: 0.00025
   62055 4197376

   prediction errors, lo: -5.9%, hi: 3.9%, mean: -0.6%, median: -0.6%


   highest absolute errors:
   period y   yhat  diff  pct 
   ---------------------------
   29     858 849.9 -8.1  -0.9
   29     817 798.6 -18.4 -2.3
   29     827 800.1 -26.9 -3.3
   29     797 750   -47   -5.9
   29     843 797   -46   -5.5

   lowest absolute errors:
   period y   yhat  diff  pct 
   ---------------------------
   18     815 811.9 -3.1  -0.4
   18     812 809   -3    -0.4
   18     773 773.6 0.6   0.1 
   18     813 802.4 -10.6 -1.3
   18     834 836.5 2.5   0.3 

   final period:
   period y   yhat  diff  pct 
   ---------------------------
   29     858 849.9 -8.1  -0.9
   29     817 798.6 -18.4 -2.3
   29     827 800.1 -26.9 -3.3
   29     797 750   -47   -5.9
   29     843 797   -46   -5.5
