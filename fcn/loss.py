"""This module provides the a softmax cross entropy loss for training FCN.

In order to train VGG first build the model and then feed apply vgg_fcn.up
to the loss. The loss function can be used in combination with any optimizer
(e.g. Adam) to finetune the whole model.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def loss(y_pred, y_true,num_classes, head=None, epsilon=0.0001):
    """Calculate the loss from the logits and the labels.

    Args:
      logits: tensor, float - [batch_size, width, height, num_classes].
          Use vgg_fcn.up as logits.
      labels: Labels tensor, int32 - [batch_size, width, height, num_classes].
          The ground truth of your data.
      head: numpy array - [num_classes]
          Weighting the loss of each class
          Optional: Prioritize some classes

    Returns:
      loss: Loss tensor of type float.
    """
    with tf.name_scope('loss'):
       y_pred = tf.reshape(y_pred, (-1, num_classes))
       shape = [y_true.get_shape()[0], num_classes]
       epsilon = tf.constant(value=epsilon, shape=shape)
       y_pred = y_pred + epsilon
       y_true = tf.to_float(tf.one_hot(y_true, depth=num_classes, axis=-1, on_value=1, off_value=0))

       softmax = tf.nn.softmax(y_pred)

       if head is not None:
           cross_entropy = -tf.reduce_sum(tf.mul(y_true * tf.log(softmax),
                                          head), reduction_indices=[1])
       else:
           cross_entropy = -tf.reduce_sum(y_true * tf.log(softmax), reduction_indices=[1])

       cross_entropy_mean = tf.reduce_mean(cross_entropy,
                                           name='xentropy_mean')
       tf.add_to_collection('losses', cross_entropy_mean)

       loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
    return loss
