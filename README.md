#METHACKS 2023 SUBMISSION - LIP READING AI
## Winning submission of Best Use of Matlab

## Usage 
To use the model, first you need to clone the repository: 
```git clone https://github.com/Chevinjeon/readinglips_c```

Then you can install the package:
```cd readinglips_c/```
```pip install -e```


## Dependencies 
* Keras 2.0+
* Tensorflow 1.0+ 
* PIP (for package installation)

![Screenshot (220)](https://github.com/Chevinjeon/readinglips_c/assets/109643560/5f84bbc7-eba7-4fb3-8830-6fa6cf4ba408)
* layout of the web application 

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

## More on the Greedy Algorithm 
* the greedy algorithm to take the most probable prediction when it comes to generating the output:
*  ``` greedy = True ```

## CTC Decoder from Tensorflow Keras
github.com/tensorflow/tensorflow/blob/r1.8/tensorflow/python/keras/impl/keras/backend.py

## end-to-end sentence-level lipreading 
Additional documentation can be found here: https://arxiv.org/abs/1611.01599

## Devpost
https://devpost.com/software/lipread-ai

## Watch the demo on localhost
https://www.youtube.com/watch?v=gd6855l01ZQ
