
.. _rnn:

Recurrent networks
==================

The `char.q <https://github.com/ktorch/examples/blob/master/rnn/char.q>`_ script implements a character-level recurrent neural network to model and predict text given a set of Shakespeare plays in
`data/shakespeare.txt <https://github.com/ktorch/examples/blob/master/rnn/data/shakespeare.txt>`_, somewhat similar to the model described `here <http://karpathy.github.io/2015/05/21/rnn-effectiveness/>`_.

Sequences of characters are used to train the network to predict the next sequence. 
After going through the data sample for 20 epochs, the trained network is used to generate text,
predicting the next character from the sequence that has come before.

::

   > q examples/rnn/char.q
   KDB+ 4.0 2020.05.04 Copyright (C) 1993-2020 Kx Systems
   l64/ 12(16)core 64037MB 

   Epoch: 1   05:37:42   training error:  2.241
   Epoch: 2   05:38:00   training error:  1.773
   Epoch: 3   05:38:18   training error:  1.644
   Epoch: 4   05:38:35   training error:  1.569
   Epoch: 5   05:38:53   training error:  1.518
   Epoch: 6   05:39:12   training error:  1.484
   Epoch: 7   05:39:30   training error:  1.458
   Epoch: 8   05:39:48   training error:  1.439
   Epoch: 9   05:40:06   training error:  1.424
   Epoch: 10  05:40:24   training error:  1.411
   Epoch: 11  05:40:42   training error:  1.400
   Epoch: 12  05:41:00   training error:  1.391
   Epoch: 13  05:41:18   training error:  1.383
   Epoch: 14  05:41:36   training error:  1.377
   Epoch: 15  05:41:54   training error:  1.370
   Epoch: 16  05:42:12   training error:  1.365
   Epoch: 17  05:42:30   training error:  1.358
   Epoch: 18  05:42:49   training error:  1.355
   Epoch: 19  05:43:07   training error:  1.351
   Epoch: 20  05:43:25   training error:  1.347

   ASTOR:
   And thou liest of his boy, and his sighs
   Prince of their false men, like a state,
   If we are breath again! a word, the glory of
   Worse than the contempt of graces: we'll be so
   more sweet a wretch, pray you, should I give me friends
   And make some contempt of noble majesty:
   So do it in a letter: there's no good hand.

   ANGELO:
   Peace, good may he is, I am parting to be reready,
   Your hearts are certain in the harm. And thou sings
   With an honest lord must draw the heart of her best
   pening lanthongers! I'll speak no ere the lodge,
   And that with so do gather in some soul of knowledge
   The noble enemy that never served the law;
   And it is pleased to speak to show the prince,
   And the bully of his new offence turns him in the reason of my troop;
   For he is this they will again to wear
   Have blood in hand good times with my esbands
   With noble right.

   AUFIDIUS:
   They, barning honour and summer, England that she stands
   By some soul of this end,
   And makes you shade such damned wive,
   This night and bur
