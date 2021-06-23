import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import activations
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import standard_ops


class DenseAP(keras.layers.Layer):
    # Based on aladdinpersson
    def __init__(self, units):
        super().__init__()
        self.units = units

    def build(self, input_shape):
        # Lazily create w and b
        self.w = (
            self.add_weight(  # Easiest way to init weights, rather than tf.variable()
                name="w",  # need to specify name to be able to save/load models
                shape=(input_shape[-1], self.units),
                initializer="glorot_uniform",
                trainable=True,
            )
        )
        self.b = self.add_weight(
            name="b",
            shape=(self.units,),
            initializer="random_normal",
            trainable=True,
            # vector of size (units,)
        )

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b


class CustomDense(keras.layers.Layer):
    # Based on nikhilroxtomar
    def __init__(self, units, activation="linear"):
        super().__init__()

        self.units = units
        self.activation = keras.layers.Activation(activation)

    def build(self, input_shape):
        # intializing weights and biases
        ## inputs = (batchsize, 784),
        # W = (784, units)
        # bias = (units)
        self.weight = self.add_weight(shape=[input_shape[-1], self.units])
        self.bias = self.add_weight(shape=[self.units])

    def call(self, input):
        # perform layer's computation
        y = tf.matmul(input, self.weight) + self.bias
        y = self.activation(y)
        return y


class Dense(keras.layers.Layer):
    """Regular densely-connected NN layer.

    Implements the operation:
    `output = activation(dot(input, kernel) + bias)`
    where `activation` is the element-wise activation function
    passed as the `activation` argument, `kernel` is a weights matrix
    created by the layer, and `bias` is a bias vector created by the layer.

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
        dtype = dtypes.as_dtype(self.dtype or K.floatx())
        if not (dtype.is_floating or dtype.is_complex):
            raise TypeError(
                "Unable to build `Dense` layer with non-floating point "
                "dtype %s" % (dtype,)
            )

        input_shape = tensor_shape.TensorShape(input_shape)
        last_dim = tensor_shape.dimension_value(input_shape[-1])
        if last_dim is None:
            raise ValueError(
                "The last dimension of the inputs to `Dense` "
                "should be defined. Found `None`."
            )
        self.input_spec = InputSpec(min_ndim=2, axes={-1: last_dim})
        # weight matrix
        self.kernel = self.add_weight(
            "kernel",
            shape=[last_dim, self.units],
            initializer=initializers.get("glorot_uniform"),
            regularizer=regularizers.get(None),
            constraint=constraints.get(None),
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
            regularizer=regularizers.get(None),
            constraint=constraints.get(None),
            dtype=self.dtype,
            trainable=True,
        )
        self.built = True

    def call(self, inputs):
        if inputs.dtype.base_dtype != self._compute_dtype_object.base_dtype:
            inputs = math_ops.cast(inputs, dtype=self._compute_dtype_object)

        rank = inputs.shape.rank
        if rank == 2 or rank is None:
            # We use embedding_lookup_sparse as a more efficient matmul operation for
            # large sparse input tensors. The op will result in a sparse gradient, as
            # opposed to sparse_ops.sparse_tensor_dense_matmul which results in dense
            # gradients. This can lead to sigfinicant speedups, see b/171762937.
            if isinstance(inputs, sparse_tensor.SparseTensor):
                # We need to fill empty rows, as the op assumes at least one id per row.
                inputs, _ = sparse_ops.sparse_fill_empty_rows(inputs, 0)
                # We need to do some munging of our input to use the embedding lookup as
                # a matrix multiply. We split our input matrix into separate ids and
                # weights tensors. The values of the ids tensor should be the column
                # indices of our input matrix and the values of the weights tensor
                # can continue to the actual matrix weights.
                # The column arrangement of ids and weights
                # will be summed over and does not matter. See the documentation for
                # sparse_ops.sparse_tensor_dense_matmul a more detailed explanation
                # of the inputs to both ops.
                ids = sparse_tensor.SparseTensor(
                    indices=inputs.indices,
                    values=inputs.indices[:, 1],
                    dense_shape=inputs.dense_shape,
                )
                weights = inputs
                outputs = embedding_ops.embedding_lookup_sparse_v2(
                    self.kernel, ids, weights, combiner="sum"
                )
            else:
                outputs = gen_math_ops.MatMul(a=inputs, b=self.kernel)
        # Broadcast kernel to inputs.
        else:
            outputs = standard_ops.tensordot(inputs, self.kernel, [[rank - 1], [0]])
            # Reshape the output back to the original ndim of the input.
            if not context.executing_eagerly():
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
