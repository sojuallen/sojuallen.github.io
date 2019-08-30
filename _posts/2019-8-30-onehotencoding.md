---
layout: post
title: A closer look at OneHotEncoding methods in Python 
---



For a while I have been doing machine learning models involved categorical variables, I kept on thinking what's the best way to do one hot encoding. While some algorithms claim to be able to take categorical as they are (eg. [lightGBM](https://lightgbm.readthedocs.io/en/latest/)), some algorithms specifically require one hot encoding, such as [logistic regresion](https://en.wikipedia.org/wiki/Logistic_regression), [SVM](https://en.wikipedia.org/wiki/Support-vector_machine). So here I summarised a few (non-exhaustive) techniques and we'll compare the pros and cons with them. 

## Construct a fake dataset



```python
import pandas as pd
import numpy as np
import scipy as sp
from sklearn.model_selection import train_test_split
pd.options.display.max_columns = 300

# creating a fake data set to start with
data = pd.DataFrame({'col_A': np.random.choice(['A','B','C','D','E'], size = 10000), 
                     'col_B': np.random.choice(['XX','YY','ZZ'], size = 10000),
                     'col_C': np.random.choice(['q','w','e','r','t','yr'], size = 10000),
                     'col_D': np.random.choice([1,2,3,4,5], size = 10000)})

# split test and train
d_train,d_test = train_test_split(data, test_size = 0.2, random_state = 200)
cat_columns = ['col_B', 'col_C', 'col_D']
```

In order to make this fake data as real as possible, I am adding a few tweaks to it: intentionally change some values in the test set that have never appeared in the train set. 


```python
d_test['col_A'] = d_test['col_A'].replace('D', 'DDD')
d_test['col_B'] = d_test['col_B'].replace('YY', 'yuyuyu')
d_test['col_C'] = d_test['col_C'].replace({'w':'what', 'yr':'year'})
```
Now let's test a few different approaches:

## OneHotEncoder from sklearn.preprocessing

ok, this name 'OneHotEncder' fooled lots of people, at least prior to sklearn version 0.19. Apparently back then people have to perform sklearn brother package 'LabelEncoder' (my god) first, then OneHotEncoder. Fortunately time does not go back and now we don't have to do that anymore in the newer version. 



```python
from sklearn.preprocessing import OneHotEncoder 
ohe = OneHotEncoder(sparse = False, handle_unknown = 'ignore')
ohe_train = ohe.fit_transform(d_train[cat_columns])
ohe_test = ohe.transform(d_test[cat_columns])
```


```python
print(ohe_train.shape, ohe_test.shape)
```

    (8000, 14) (2000, 14)


the above result looks fine I guess, columns are matching between train and test, I mean it is doing what you asked it to do. But a closer look at it I found:
* it returns a matrix with no column names
* it only did transformation on the cat_columns I appointed (of course) 
This means there are a few extra steps you have to do to get it back to a nice, complete data frame (obtain and reassign column names, merge with the columns that are not in this transformation...) 


## category_encoder package 

see here [category_encoder](http://contrib.scikit-learn.org/categorical-encoding/index.html) for more information. 
The beauty of this package is it can handle a bunch of additional transformation tasks other than OneHotEncoding. ~wait, I only care about OneHotEncoding here~ such as: Backward Difference Coding, BaseN, Binary, CatBoost Encoder, Hashing, Helmert Coding, James-Stein Encoder, Leave One Out, M-estimate, One Hot, Ordinal, Polynomial,Coding, Sum Coding, Target Encoder, Weight of Evidence. It does made it easier by centralising lots of these tasks. Now let's test it:


```python
import category_encoders as ce
ce_ohe = ce.OneHotEncoder(handle_unknown = 'ignore', use_cat_names = True)
train_ce = ce_ohe.fit_transform(d_train[cat_columns])
test_ce = ce_ohe.transform(d_train[cat_columns])    
```


```python
print(train_ce.shape, test_ce.shape)
```

    (8000, 10) (8000, 10)


The results also seem to be a little weird, there is a problem here: 
in the dataset, I intentionally created 'col_D' to be numerical. It turned out that since python recognised the column is numerical, the encoder did not touch the column at all. 


```python
train_ce.head()
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>col_B_ZZ</th>
      <th>col_B_YY</th>
      <th>col_B_XX</th>
      <th>col_C_q</th>
      <th>col_C_e</th>
      <th>col_C_r</th>
      <th>col_C_w</th>
      <th>col_C_t</th>
      <th>col_C_yr</th>
      <th>col_D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5541</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>5408</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2364</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1257</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6342</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>




In addition, in real life people still have to do one extra step: 
put the non-categorical variables back. 
Sure, we can go ahead and do a **pd.concat** for the job, but we have to keep reminding ourselves that. 

## get_dummies
Coming back to the old fashioned get_dummies. quickly I realised that the function itself doesn't do the whole job neither: it only give you a raw one hot encoded table, with no rule, no method to be passed to test set. But one good thing is, it will always do one hot encoding regardless of the column type (ie. numerical, object..etc). That means people don't need to specify the col_types of column anymore, in comparison of the previous method. 


```python
pd.get_dummies(d_train, prefix_sep = '__', columns = cat_columns).head()
```




<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>col_A</th>
      <th>col_B__XX</th>
      <th>col_B__YY</th>
      <th>col_B__ZZ</th>
      <th>col_C__e</th>
      <th>col_C__q</th>
      <th>col_C__r</th>
      <th>col_C__t</th>
      <th>col_C__w</th>
      <th>col_C__yr</th>
      <th>col_D__1</th>
      <th>col_D__2</th>
      <th>col_D__3</th>
      <th>col_D__4</th>
      <th>col_D__5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5541</th>
      <td>B</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5408</th>
      <td>B</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2364</th>
      <td>B</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1257</th>
      <td>D</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6342</th>
      <td>E</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>




## Seems like there is no free lunch

Well unfortunately it does not seem to have a nice & easy one liner code that can help to do my job ~so I can lie on the couch~. Let's create a function then. 

![alt text](https://raw.githubusercontent.com/sojuallen/sojuallen.github.io/master/images/thinkingkid.jpg)

```python
def ohe_train_test(train, test, cat_columns = cat_columns): # this one is slightly faster
    '''handle the train and test one hot encoding consistent to the train category levels'''
    train_ec = pd.DataFrame(pd.get_dummies(train, prefix_sep = '__', columns = cat_columns)) 
    test_ec = pd.DataFrame(pd.get_dummies(test, prefix_sep = '__', columns = cat_columns))
    # algin two df
    train_df, test_df = train_ec.align(test_ec, join ='left', axis = 1, fill_value = 0)
    return train_df, test_df
```


```python
ec_train, ec_test = ohe_train_test(d_train, d_test) 
```


```python
ec_train.head()
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>col_A</th>
      <th>col_B__XX</th>
      <th>col_B__YY</th>
      <th>col_B__ZZ</th>
      <th>col_C__e</th>
      <th>col_C__q</th>
      <th>col_C__r</th>
      <th>col_C__t</th>
      <th>col_C__w</th>
      <th>col_C__yr</th>
      <th>col_D__1</th>
      <th>col_D__2</th>
      <th>col_D__3</th>
      <th>col_D__4</th>
      <th>col_D__5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5541</th>
      <td>B</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5408</th>
      <td>B</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2364</th>
      <td>B</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1257</th>
      <td>D</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6342</th>
      <td>E</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>



great! nicely done. A few things: 
* col_A is already there in the result dataframe, so I don't have to think about it.
* all columns are aligned between train and test
* result is nicely in dataframe format with all column names. 

Problem solved! 

## what about speed?


```python
from timeit import timeit

# for OneHotEncoder from sklearn.preprocessing
%timeit ohe_train = ohe.fit_transform(d_train[cat_columns])
%timeit ohe_test = ohe.transform(d_test[cat_columns])
```

    9.14 ms ± 359 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
    4.25 ms ± 544 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)


```python
# for category_encoder
%timeit train_ce = ce_ohe.fit_transform(d_train[cat_columns])
%timeit test_ce = ce_ohe.transform(d_train[cat_columns])
```

    22.6 ms ± 668 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
    20.3 ms ± 642 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)



```python
# for my pre-defined function with get_dummies
%timeit ohe_train_test(d_train, d_test)
```

    15.2 ms ± 674 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)


Well my defined function is pretty fast compared to the other two options. great! 
