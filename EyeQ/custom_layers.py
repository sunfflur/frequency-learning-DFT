
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Layer

### frequency layer constrution ###

#@tf.keras.utils.register_keras_serializable('Custom')


### frequency layer constrution ###

class FreqLayer(Layer):
  def __init__(self, units, kernel_initializer='RandomNormal', extra_layers='Dense', output_type='Dense', **kwargs):
    super().__init__(**kwargs)
    self.units = units
    self.kernel_initializer = kernel_initializer
    self.extra_layers = extra_layers
    self.output_type = output_type
  def get_config(self):
    config = super(FreqLayer, self).get_config()
    config.update({
      "units": self.units,
      "kernel_initializer": self.kernel_initializer,
      "extra_layers": self.extra_layers,
      "output_type": self.output_type
      })
    return config
  def build(self, batch_input_shape):
    if self.extra_layers == 'Dense':
      shape = [batch_input_shape[-1]] #(weights,) - shape for dense training
    else:
      shape = [batch_input_shape[-2],1] #(weights, 1) - shape for convolutional training
    self.kernel = self.add_weight(
        name='kernel',
        shape = shape,
        initializer=self.kernel_initializer,
        trainable=True)
    super().build(batch_input_shape) # must be at the end
  def call(self, X):
    #### fourier convolution ####
    f = X*self.kernel

    #### hartley convolution ####
    #f = conv(X, self.kernel)

    return f
  def compute_output_shape(self, batch_input_shape):
    if self.output_type == 'Dense':
      return tf.TensorShape(batch_input_shape.as_list()[:-1] + [self.units]) #[examples,1] - dense
    else:
      return tf.TensorShape(batch_input_shape.as_list()[:-2] + [self.units]) #[examples,1] - conv1d


#pass

#assert tf.keras.utils.get_registered_object('Custom>FreqLayer') == FreqLayer
#assert tf.keras.utils.get_registered_name(FreqLayer) == 'Custom>FreqLayer'

'''
  def get_config(self):
    config = super().get_config()
    config["units"] = self.units,
    config["kernel_initializer"] = self.kernel_initializer,
    config["extra_layers"] = self.extra_layers,
    config["output_type"] = self.output_type
    return config
'''


'''
  def get_config(self):
    config = super(FreqLayer, self).get_config()
    config.update({
      "units": self.units,
      "kernel_initializer": self.kernel_initializer,
      "extra_layers": self.extra_layers,
      "output_type": self.output_type
      })
    return config
'''
  

"""#### Custom Layer"""
'''
### 1D DISCRETE HARTLEY TRANSFORM FUNCTIONS ###

def dht(x):
  x = tf.cast(x, tf.complex128)
  X = tf.signal.fft(x)
  X = tf.math.real(X)-tf.math.imag(X)
  return X

def idht(X):
  # Compute the IDHT for a sequence x of length n using the FFT. 
  # Since the DHT is involutory, IDHT(x) = 1/n DHT(H) = 1/n DHT(DHT(x))

  n = X.shape[1] #number of coefs
  x = dht(X)
  x = (1.0/n)*x
  return x

def flipx(X):
  x = tf.experimental.numpy.flip(X, axis=1)
  return x

def flipy(Y):
  y = tf.experimental.numpy.flip(Y, axis=0)
  return y


def conv(X, y):

  # Computes the DHT of the convolution of x and y, sequences of length n, using FFT.

  #X = dht(X) # my input is already the DHT of x
  #X = tf.cast(X, tf.float64)
  #Y = tf.cast(dht(y), tf.float32) # kernel
  Y = y
  X_flip = flipx(X)
  Y_flip = flipy(Y)

  #X_even = 0.5 * (X + X_flip)
  #X_even_flip = flipx(X_even)

  Y_even = 0.5 * (Y + Y_flip)
  Y_odd = 0.5 * (Y - Y_flip)

  Z = X*Y_even + X_flip*Y_odd

  #Z = X_even*Y
  #Z =   X*Y_even
  #Z = X*dht(Y_even)
  
  # Once the DHT of the convolution has been computed using the DHT of the conv, computing the convolution just requires a IDHT.
  z = Z # DHT of the convolution
  #z = idht(Z) # convolution
  return z
'''
