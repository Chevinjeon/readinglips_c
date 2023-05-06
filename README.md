# LIP READING AI


### Non-Batched 
```python
image = [[[1,2,3],[4,5,6],[7,8,9]]...]
#Adds an extra axis, in effect batches 
batched = tf.expand_dims(image, axis=0)
```

### Batched
```python 
batched = [[[[1,2,3],[4,5,6],[7,8,9]]...]]
```

### CTC Decoder from Tensorflow Keras
Additional documentation on end-to-end sentence-level lipreading can be found here: https://arxiv.org/abs/1611.01599
