--
layout: post
title: A summary of some regression techniques with Python code
categories: [linear models, python]
tags: [regression, ridge, lasso, elastic net, python]
--



## Linear Regression: 

cost function:
$${1\over 2m}\sum_{i=1}^m(h_\theta(x^{(i)})-y^{(i)})^2 $$



```python
# Pseudo code for LR
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)
lr.predict(X_valid)
```

## Ridge:

cost function: 
$$min(||Y-X \theta||_2^2 + \lambda||\theta||_2^2)$$
* It shrinks the parameters, therefore it is mostly used to prevent multicollinearity.
* It reduces the model complexity by coefficient shrinkage.
* It uses L2 regularization technique. 

note:
the double absolute notation means the Euclidean norm of the vector a. or 'distance', ie. $$||a|| = \sqrt{a_x^2+a_y^2+a_z^2}$$


```python
# Pseudo code for Ridge
from sklearn.linear_model import Ridge 
rr = Ridge(alpha = 0.5, normalize = True)
rr.fit(X_train, y_train)
rr.predict(X_valid)
```

## Lasso

cost function:
$$min(||Y-X \theta||_2^2 + \lambda||\theta||_1)$$
* It uses L1 regularization technique
* It is generally used when we have more number of features, as it does feature selection by reducing some features variables to 0.


```python
# Pseudo code for Lasso
from sklearn.linear_model import Lasso 
lasr = Lasso(alpha = 0.3, normalize = True)
lasr.fit(X_train, y_train)
lasr.predict(X_valid)
```

## Elastic net: (hybrid of ridge & lasso)

cost function:
$$min(||Y-X \theta||_2^2 + \lambda||\theta||_1 + \lambda||\theta||_2^2)$$
Elastic net can be understood from below quote:
> You are trying to catch a fish from a pond. And you only have a net, then what would you do? Will you randomly throw your net? No, you will actually wait until you see one fish swimming around, then you would throw the net in that direction to basically collect the entire group of fishes. Therefore even if they are correlated, we still want to look at their entire group.

Elastic net is controlled by 2 variables: Alpha and L1_ratio:
$$ alpha = a + b,  L_1 ratio = {a\over {a+b}} $$
the trade off between L1 and L2 is: $$a*(L_1 term)+b*(L_2 term)$$
let alpha = (a+b) = 1:
* if l1_ratio = 1 = a/(a+b) -> a = 1, so b = 0 -> Lasso 
* if l1_ratio = 0, -> a = 0 -> Ridge 
* if 0 < l1_ratio < 1 -> penalty is the combination of ridge and lasso 


```python
from sklearn.linear_model import ElasticNet
en = ElasticNet(alpha = 1, l1_ratio = 0.5, normalize = False) 
en.fit(X_train, y_train)
en.predict(X_valid)
```
