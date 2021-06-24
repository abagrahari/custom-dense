import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.eager import context
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import activations
from tensorflow.python.keras import initializers
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import standard_ops


class Dense(keras.layers.Layer):
    """Regular densely-connected NN layer. Simplified from TensorFlow implementation.

    Implements the operation:
    `output = activation(dot(input, kernel) + bias)`
    where `activation` is the element-wise activation function
    passed as the `activation` argument, `kernel` is a weights matrix
    created by the layer, and `bias` is a bias vector created by the layer.
    The dot product is performed on the last dimension of the input.

    Args:
      units: Positive integer, dimensionality of the output space.
      activation: Activation function to use.
        If you don't specify anything, no activation is applied
        (ie. "linear" activation: `a(x) = x`).

    Input shape:
      N-D tensor with shape: `(batch_size, ..., input_dim)`.
      The most common situation would be
      a 2D input with shape `(batch_size, input_dim)`.
    Output shape:
      N-D tensor with shape: `(batch_size, ..., units)`.
      For instance, for a 2D input with shape `(batch_size, input_dim)`,
      the output would have shape `(batch_size, units)`.
    """

    def __init__(
        self,
        units: int,
        activation=None,
        **kwargs,
    ):

        super().__init__(**kwargs)

        self.units = units
        self.activation = activations.get(activation)
        self.input_spec = InputSpec(min_ndim=2)

    def build(self, input_shape):
        # Lazily create and init weights and biases, since shape of input tensor is now known
        # inputs.shape is (batchsize, features)
        # kernel.shape is (features, units)
        # bias.shape is (units)
        if input_shape[-1] is None:
            raise ValueError(
                "The last dimension of the inputs to `Dense` "
                "should be defined. Found `None`."
            )
        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_shape[-1]})
        # weight matrix
        self.kernel = self.add_weight(
            "kernel",  # need to specify name to be able to save/load models
            shape=[input_shape[-1], self.units],
            initializer=initializers.get("glorot_uniform"),
            dtype=self.dtype,
            trainable=True,
        )
        # bias vector
        self.bias = self.add_weight(
            "bias",
            shape=[
                self.units,
            ],
            initializer=initializers.get("zeros"),  # for bias vector
            dtype=self.dtype,
            trainable=True,
            # vector of size (units,)
        )
        self.built = True

    def call(self, inputs):
        # perform layer's computation, in the forward pass.
        # Back prop is automatically handled by tf
        # Can only use tensorflow functions here
        if inputs.dtype.base_dtype != self._compute_dtype_object.base_dtype:
            inputs = math_ops.cast(inputs, dtype=self._compute_dtype_object)

        rank = inputs.shape.rank
        if rank == 2 or rank is None:
            outputs = gen_math_ops.MatMul(a=inputs, b=self.kernel)
            # outputs = tf.matmul(inputs, self.kernel)
        else:
            # Broadcast kernel to inputs.
            outputs = standard_ops.tensordot(inputs, self.kernel, [[rank - 1], [0]])
            # Reshape the output back to the original ndim of the input.
            shape = inputs.shape.as_list()
            output_shape = shape[:-1] + [self.kernel.shape[-1]]
            outputs.set_shape(output_shape)

        outputs = nn_ops.bias_add(outputs, self.bias)

        if self.activation is not None:
            outputs = self.activation(outputs)

        return outputs

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        input_shape = input_shape.with_rank_at_least(2)
        if tensor_shape.dimension_value(input_shape[-1]) is None:
            raise ValueError(
                "The innermost dimension of input_shape must be defined, but saw: %s"
                % (input_shape,)
            )
        return input_shape[:-1].concatenate(self.units)
