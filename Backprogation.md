

```python
import numpy as np

```


```python
X=np.array([[1,0,1,1],[1,0,0,1],[1,0,1,0],[1,1,1,0]])
print(X)
Y=np.array([1,1,0,1])
print(Y)
```

    [[1 0 1 1]
     [1 0 0 1]
     [1 0 1 0]
     [1 1 1 0]]
    [1 1 0 1]
    


```python
wh= np.random.random((4,4))
bh= np.random.random((1,4))
#**Weights**
print("WEIGHTS \n",wh)
#**Bais**
print("\nBIAS\n",bh)
```

    WEIGHTS 
     [[0.7395757  0.4892827  0.29152561 0.07168798]
     [0.9053082  0.91268534 0.36623927 0.99279   ]
     [0.0727625  0.47649619 0.89463153 0.54878761]
     [0.0370883  0.93403931 0.09342449 0.30304004]]
    
    BIAS
     [[0.50854475 0.6249031  0.16258624 0.59844861]]
    


```python
wout = np.random.uniform(low=0, high=1, size=(4,1))
bout = np.random.uniform(low=0, high=1, size=(1,1))
#**Weights**
print("WOUT \n",wout)
#**Bais**
print("\n BOUT \n",bout)
```

    WOUT 
     [[0.98530159]
     [0.59401434]
     [0.2804366 ]
     [0.01233856]]
    
     BOUT 
     [[0.27580602]]
    


```python
hidden_layer_input = X.dot(wh) + bh
print("\n Hidden Layer \n",hidden_layer_input)
```

    
     HIdden Layer [[1.35797126 2.5247213  1.44216786 1.52196424]
     [1.28520876 2.04822511 0.54753633 0.97317663]
     [1.32088296 1.590682   1.34874337 1.2189242 ]
     [2.22619116 2.50336733 1.71498264 2.2117142 ]]
    


```python
 # activation function
  def sigmoid (x): return 1/(1 + np.exp(-x))  
  hiddenlayer_activations = sigmoid(hidden_layer_input)
  print("\n hiddenlayer_activations \n",hiddenlayer_activations)
```

    
     hiddenlayer_activations 
     [[0.79542978 0.92585681 0.80879013 0.82082754]
     [0.78333512 0.88576815 0.63356381 0.72575222]
     [0.78932857 0.83071203 0.79392411 0.77187417]
     [0.90257695 0.92437755 0.84748144 0.90129653]]
    


```python
output_layer_input = hiddenlayer_activations.dot(wout ) + bout 
print("\n output_layer_input \n",output_layer_input)
output=sigmoid(output_layer_input)
print("\n output \n",output)

  
```

    
     output_layer_input 
     [[1.84645865]
     [1.76041556]
     [1.77915677]
     [1.96299556]]
    
     output 
     [[0.86371077]
     [0.8532617 ]
     [0.85559271]
     [0.87685678]]
    


```python
error=Y-output
print("\n error \n",error)

```

    
     error 
     [[ 0.13628923  0.13628923 -0.86371077  0.13628923]
     [ 0.1467383   0.1467383  -0.8532617   0.1467383 ]
     [ 0.14440729  0.14440729 -0.85559271  0.14440729]
     [ 0.12314322  0.12314322 -0.87685678  0.12314322]]
    


```python
# derivative of sigmoid
def derivatives_sigmoid(x): return x * (1 - x)             
Slope_output_layer= derivatives_sigmoid(output)
Slope_hidden_layer = derivatives_sigmoid(hiddenlayer_activations)
print("\n Slope_output_layer \n",Slope_output_layer)
print("\n Slope_hidden_layer \n",Slope_hidden_layer)
```

    
     Slope_output_layer 
     [[0.11771447]
     [0.12520617]
     [0.12355382]
     [0.10797897]]
    
     Slope_hidden_layer 
     [[0.16272125 0.06864598 0.15464865 0.14706969]
     [0.16972121 0.10118293 0.23216071 0.19903594]
     [0.16628898 0.14062955 0.16360862 0.17608443]
     [0.0879318  0.0699037  0.12925665 0.0889611 ]]
    


```python
d_output = error * Slope_output_layer
print("\n d_output \n",d_output)

```

    
     d_output 
     [[ 0.01604322  0.01604322 -0.10167126  0.01604322]
     [ 0.01837254  0.01837254 -0.10683363  0.01837254]
     [ 0.01784207  0.01784207 -0.10571175  0.01784207]
     [ 0.01329688  0.01329688 -0.09468209  0.01329688]]
    


```python
Error_at_hidden_layer = d_output.T.dot(wout)
print("\n Error_at_hidden_layer \n",Error_at_hidden_layer)


```

    
     Error_at_hidden_layer 
     [[ 0.03188859]
     [ 0.03188859]
     [-0.19445125]
     [ 0.03188859]]
    


```python
d_hiddenlayer = Error_at_hidden_layer * Slope_hidden_layer
print("\n d_hiddenlayer \n",d_hiddenlayer)
```

    
     d_hiddenlayer 
     [[ 0.00518895  0.00218902  0.00493153  0.00468985]
     [ 0.00541217  0.00322658  0.00740328  0.00634698]
     [-0.0323351  -0.02734559 -0.0318139  -0.03423984]
     [ 0.00280402  0.00222913  0.00412181  0.00283684]]
    


```python
wout= wout+hiddenlayer_activations.T.dot(d_output)
wh= wh+X.T.dot(d_hiddenlayer)
print("\n updated weight at output layer \n",wout)
print("\n updated weight at hidden layer \n",wh)


```

    
     updated weight at output layer 
     [[ 1.14501505  1.14501505 -0.0150726   1.14501505]
     [ 0.76873581  0.76873581 -0.4982883   0.76873581]
     [ 0.43058624  0.43058624 -0.66181857  0.43058624]
     [ 0.16911523  0.16911523 -0.97142782  0.16911523]]
    
     updated weight at hidden layer 
     [[0.70171579 0.44988099 0.26081104 0.03095564]
     [0.91091625 0.9171436  0.37448289 0.99846369]
     [0.02407825 0.43064132 0.84911041 0.49536131]
     [0.05829055 0.94487052 0.1180941  0.32511368]]
    


```python
bh = bh +np.sum(d_hiddenlayer, axis=0) 

bout = bout + np.sum(d_output, axis=0)

print("\n updated bias at output layer \n",bout)
print("\n updated bias at hidden layer \n",bh)


```

    
     updated bias at output layer 
     [[ 0.40691544  0.40691544 -0.54199145  0.40691544]]
    
     updated bias at hidden layer 
     [[0.47068484 0.58550139 0.13187168 0.55771627]]
    
