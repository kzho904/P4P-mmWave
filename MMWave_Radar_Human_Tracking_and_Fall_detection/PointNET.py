import tensorflow as tf
from tensorflow import keras
from keras import ops, layers
import numpy as np

keras.utils.set_random_seed(seed=42)

# Data augmentation function
def augment(points, label):
    # Jitter points
    points += tf.random.uniform(points.shape, -0.005, 0.005, dtype="float16")
    # Shuffle points
    points = tf.random.shuffle(points)
    return points, label

# Convolutional block with batch normalization and ReLU activation
def conv_bn(x, filters):
    x = layers.Conv2D(filters, kernel_size=(1, 1), padding="valid")(x)
    x = layers.BatchNormalization(momentum=0.0)(x)
    return layers.Activation("relu")(x)

# Dense block with batch normalization and ReLU activation
def dense_bn(x, filters):
    x = layers.Dense(filters)(x)
    x = layers.BatchNormalization(momentum=0.0)(x)
    return layers.Activation("relu")(x)

# Transformation Network (T-Net) for feature transformation
def tnet(inputs, num_features):
    # Initialize bias as the identity matrix
    bias = keras.initializers.Constant(np.eye(num_features).flatten())
    reg = OrthogonalRegularizer(num_features)

    # Convolutional layers using Conv2D
    x = conv_bn(inputs, 32)
    x = conv_bn(x, 64)
    x = conv_bn(x, 512)

    # GlobalMaxPooling2D instead of GlobalMaxPooling1D
    x = layers.GlobalMaxPooling2D()(x)

    # Dense layers
    x = dense_bn(x, 256)
    x = dense_bn(x, 128)
    x = layers.Dense(
        num_features * num_features,
        kernel_initializer="zeros",
        bias_initializer=bias,
        activity_regularizer=reg,
    )(x)

    # Reshape and apply transformation
    feat_T = layers.Reshape((num_features, num_features))(x)
    return layers.Dot(axes=(2, 1))([inputs, feat_T])

# Custom regularizer to encourage orthogonal transformation matrices
class OrthogonalRegularizer(keras.regularizers.Regularizer):
    def __init__(self, num_features, l2reg=0.001):
        self.num_features = num_features
        self.l2reg = l2reg
        self.eye = ops.eye(num_features)

    def __call__(self, x):
        x = ops.reshape(x, (-1, self.num_features, self.num_features))
        xxt = ops.tensordot(x, x, axes=(2, 2))
        xxt = ops.reshape(xxt, (-1, self.num_features, self.num_features))
        return ops.sum(self.l2reg * ops.square(xxt - self.eye))