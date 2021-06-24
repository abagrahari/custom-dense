# custom-dense
Creating a custom dense layer

## Environment setup:
- `conda env create --file environment.yml`
- `conda activate tf-exp`

## Docs
- [API Docs of Dense layer](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense)
- [Dense layer implementation in TensorFlow](https://github.com/tensorflow/tensorflow/blob/5697e8d298073398ed472bba788c4b6a7bd980d1/tensorflow/python/keras/layers/core.py#L1089)
- [Explanation on parameters of the Dense layer](https://machinelearningknowledge.ai/keras-dense-layer-explained-for-beginners/ )
- [The math behind neural networks - Analysing a Dense Layer](https://vallant.in/the-math-behind-neural-networks-analysing-a-dense-layer/?__cf_chl_managed_tk__=b705b0bee1c6386a5c42ef1ad4d2409fd7eacb67-1624480399-0-AVBxDceeqOjJkuord56P2Tyk6aLo9W9lS4HWfF4rS4JpjAo2XKns76ZY1CjY8bItZeD_EJyqKMR_s_v56DqG_gR778v3g_7zmkGyu1Iy66YWVufasY70MfbrsWtpU5iUiL_liYa0KDLwSlGT9gupkymzHuokExyHFxFFNQwWsQ_hTy-zvjNEPtnAKe0Stt7VPuXXpv-UkEye3TUiBbRfIkVyRztXro5lnuB88-c1yAySt-r0JCSqYZY1lN7KFE64dINiyoUHQsJx1PQrzULxZVCzG4iaWrf7zkDbLT5NkaFpeSBhGGYBIxSIKwrqBF9N-YbGvHLNN-QRmkv4r4B7u3x8i1qNu3O669HehLnVVuwoYqJ1fpYPZeigd6TK3zRrcMEbYnIf3ma8_jOv_2MrPV3AXDJgEqUc_KZwT-1qYIxoFgkHYYkGtgo6_iboik5jpwBZgm_0-Lv6AEj7UdFSsFhfIzfXtJ2hW3p11c6Tz7ssQRiGGGygI47ILNIiFIKVqTgqAdbCPu5EqMfqini179vuTOwuL_qE1Tl6Hu3hYrGKp1iOWpKLwvN8D1tQ8WddLcpMVuYpt6cDScgxD_Z1ny-ckF759vDBx1DAz1VDOkBKqrXBoyuNewbJ-clygiYrVKYWkDCaIeEmA5GimqM4UMRdqZsSddhvCHsphlmZ8xCw0ptMRXMzFnQwT5kEFUPabYndG95Wv_hmLY580DcOap49sQPccTbsJsXmvSs7oqkjVJHFhoa6b-AipKi8wc_NOw)

## Custom layers in TensorFlow
- [Youtube - Tutorial - Custom Layers](https://youtu.be/cKMJDkWSDnY?t=235)
    - [Link to code](https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/TensorFlow/Basics/tutorial9-custom-layers.py)
- [Youtube - Custom Dense Layer in TensorFlow](https://youtu.be/0jhdNS59SdU?t=367)
     - [Code](https://github.com/nikhilroxtomar/Custom-Layer-in-TensorFlow-using-Keras-API/blob/main/custom.py)
- https://sparrow.dev/keras-custom-layer/
- [TowardsDataSciecne - How to make a custom TensorFlow 2 layer](https://towardsdatascience.com/im-out-of-the-layers-how-to-make-a-custom-tensorflow-2-layer-9921942c88fc)
- https://www.tensorflow.org/guide/keras/custom_layers_and_models
- https://www.tensorflow.org/tutorials/customization/custom_layers
- https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer

## Math
Dense layer implements `y = W*x+b`