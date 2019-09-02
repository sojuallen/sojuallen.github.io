---
layout: post
title: fastest way to combine columns in Python 
---

What’s the fastest way to combine columns? Let’s test it:

let’s create a fake data set (about 500k rows), with a few string columns, and let’s test it

```python
import pandas as pd
import numpy as np
import scipy as sp

# creating a fake data set (500k rows) to start with
data = pd.DataFrame({'col_A': np.random.choice(['Aasdfs234234234adf','B234wer234234fd','C12dfscsd234234fs','D123sdcs234235cx','Ekjghjn234543rfgn'], size = 500000), 
                     'col_B': np.random.choice(['XXJKDSHFKJNSDKFNSKDFSDFSDFSDFSDF','YYSDFCCCCXCVXCVDSFSFSDFSDSSSDF','ZZS#@4EFDRGSERW#REFD'], size = 500000),
                     'col_C': np.random.choice(['qawerq3','w234rfgf','e234gbnjy','r234rfgdfs','234rfdgt','yyhbnjuy654err'], size = 500000)})

m_cols = ['col_A', 'col_C']

###################################################
## Now test it#####################################

# Option 1: apply a lambda function to join strings
%timeit data[m_cols].apply(lambda x:''.join(x), axis = 1)
# 8.01 s ± 303 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

# Option 2: pandas cat: 
%timeit data['col_A'].str.cat(data['col_C'])
# 139 ms ± 4.27 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)

# Option 3: using sum to concatenate strings
%timeit data[m_cols].sum(axis =1)
# 215 ms ± 10.3 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

# Option 4:  Just add the two columns (if you know they are strings)
%timeit data['col_A']  + data['col_C']
# 65.1 ms ± 2.44 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)

```

*Conclusion*:

Option 1 is very popular right out there, but it is the slowest. Option 4 string addition approach is the fastest. It is 8010/65.1 = 123 times faster!!! 
