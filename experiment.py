"""Comparing different fake_quant related operations available in the tf package."""
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf

SEED = 3
tf.random.set_seed(SEED)


def CompareQuantize(x, r):
    print(r)
    print("Before ", x)
    unused_val = 0
    y = tf.quantization.quantize_and_dequantize_v2(
        x, unused_val, unused_val, range_given=False
    )
    z = tf.quantization.quantize(x, -r, r, tf.dtypes.qint8, mode="SCALED")
    x = tf.quantization.fake_quant_with_min_max_vars(x, -r, r)

    print("After y ", y)
    print("After z ", z)
    print("After x ", x)

    print()
    return x


x = tf.random.uniform(shape=[1, 8], maxval=3, dtype=tf.float32, seed=SEED)
x = CompareQuantize(x, 10.0)
x = x + 5
x = CompareQuantize(x, 15.0)
x = x - 5
x = CompareQuantize(x, 10.0)
x = x * 0.1
x = CompareQuantize(x, 1.0)
w = tf.random.uniform(shape=[8, 1], maxval=5, dtype=tf.float32, seed=SEED)
x = tf.matmul(x, w)
x = CompareQuantize(x, 10.0)

import sys

sys.exit(0)
