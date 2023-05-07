# LIP READING AI

## Dependencies 
* Keras 2.0+
* Tensorflow 1.0+ 
* PIP (for package installation)



## Note
* ```model.predict()``` expects a batch of inputs. We only have one input that we are going to be passing through our model so we need to wrap it inside of another set of arrays.
  * For example, ```yhat = model.predict(tf.expand_dims(video, axis=0))``` will do this relatively easily. 
  * This will return our predictions, which we run through the Keras CTC Decoder.

## Non-Batched 
```
image = [[[1,2,3],[4,5,6],[7,8,9]]...]
#Adds an extra axis, in effect batches 
batched = tf.expand_dims(image, axis=0)
```

## Batched
``` 
batched = [[[[1,2,3],[4,5,6],[7,8,9]]...]]
```

## Greedy 
* the greedy algorithm to take the most probable prediction when it comes to generating the output:
*  ``` greedy = True ```

## CTC Decoder from Tensorflow Keras
github.com/tensorflow/tensorflow/blob/r1.8/tensorflow/python/keras/impl/keras/backend.py

## end-to-end sentence-level lipreading 
Additional documentation can be found here: https://arxiv.org/abs/1611.01599
