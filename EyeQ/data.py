
import tensorflow as tf
import numpy as np


"""#### Data Load"""

### Saved Pre-Processed data LOAD ###

def Load(path_train=None, y_train=None, path_valid=None, y_valid=None, path_test=None, y_test=None, extra_layers='Dense'):
  y_train, y_test = tf.stack(np.load(y_train)), tf.stack(np.load(y_test))
  if extra_layers=='Dense':
    if path_valid!=False:
      input_train = tf.convert_to_tensor(np.load(path_train)) ### train 
      input_valid = tf.convert_to_tensor(np.load(path_valid)) ### valid
      input_test = tf.convert_to_tensor(np.load(path_test)) ### test
      return input_train, y_train, input_valid, y_valid, input_test, y_test
    else:
      input_train = tf.convert_to_tensor(np.load(path_train)) ### train 
      input_test = tf.convert_to_tensor(np.load(path_test)) ### test
      return input_train, y_train, input_test, y_test
  else:
    if path_valid!=False:
      input_train = tf.convert_to_tensor(np.load(path_train)) ### train 
      input_train = tf.expand_dims(input_train, -1)
      input_valid = tf.convert_to_tensor(np.load(path_valid)) ### valid
      input_valid = tf.expand_dims(input_valid, -1)
      input_test = tf.convert_to_tensor(np.load(path_test)) ### test
      input_test = tf.expand_dims(input_test, -1)
      return input_train, y_train, input_valid, y_valid, input_test, y_test
    else:
      input_train = tf.convert_to_tensor(np.load(path_train)) ### train 
      input_train = tf.expand_dims(input_train, -1)
      input_test = tf.convert_to_tensor(np.load(path_test)) ### test
      input_test = tf.expand_dims(input_test, -1)
      return input_train, y_train, input_test, y_test
  #return input_train, input_valid, input_test


def input_split(x=None, image_shape=None, levels=3, ch=1):
  coefs1 = int((image_shape/2)/2) #
  #ch = 3 #channels

  # layer 3 (global)
  C = x[:, 0:coefs1*ch] #TensorShape([3360, 384])

  # layer 2
  B = x[:, coefs1*ch:(coefs1*ch)+(2*coefs1*ch)] #TensorShape([3360, 768])

  # layer 1 (local)
  A = x[:, (coefs1*ch)+(2*coefs1*ch):(coefs1*ch)+(2*coefs1*ch)+(4*coefs1*ch)] #TensorShape([3360, 1536]) 

  return A, B, C
