"""

"""

# internal python imports
import os

# third party
import numpy as np
import tensorflow as tf
from tensorflow import keras as keras
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer

# local utils
from . import utils


class JacobianMultiplyIntensities(keras.layers.Layer):

    def __init__(self, indexing='ij', **kwargs):
        self.indexing = indexing
        super(self.__class__, self).__init__(**kwargs)
  
    def build(self, input_shape):
        self.inshape = input_shape

    def call(self, inputs):
        """
        input : [loc_shift, moved_image]
        output : Moved image with intensities multiplied by Jacobian determinant of the transformation.
        """
        _, jacTransfo = utils.jacobian(utils.shift_to_transfo(inputs[1], indexing=self.indexing), outDet=True)
        jacTransfo = tf.math.abs(jacTransfo)        
        
        return tf.expand_dims(jacTransfo,-1) * inputs[0]


class Slice(keras.layers.Layer):

    def __init__(self, index=0, **kwargs):
        self.index = index
        super(self.__class__, self).__init__(**kwargs)
  
    def build(self, input_shape):
        self.inshape = input_shape

    def call(self, inputs):
        """
        input : [index]
        output : Extract indexth slice along last dimension.
        """  
        output = tf.expand_dims(inputs[..., self.index],-1)
        
        return output

