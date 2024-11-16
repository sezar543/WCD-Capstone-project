import tensorflow as tf
from tensorflow import keras
from keras import backend as K
import gc

gc.collect()  # Manually collect garbage

# Defining a function that calculates the F-beta score for a given set of 
# #true labels and predicted labels.
# The function balances precision and recall and it is useful when there is 
# an imbalance in the number of positive and negative examples in the data.

# https://www.tensorflow.org/tutorials/keras/save_and_load
#@tf.function
@keras.saving.register_keras_serializable(name="fbeta")
def fbeta(y_true, y_pred, threshold_shift=0):
    beta = 2

    # Clipping y_pred between 0 and 1
    y_pred = K.clip(y_pred, 0, 1)

    # Rounding y_pred to binary values
    y_pred_bin = K.round(y_pred + threshold_shift)

    # Counting true positives, false positives, and false negatives
    tp = K.sum(K.round(y_true * y_pred_bin)) + K.epsilon()
    fp = K.sum(K.round(K.clip(y_pred_bin - y_true, 0, 1)))
    fn = K.sum(K.round(K.clip(y_true - y_pred, 0, 1)))

    # Calculating precision and recall
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    beta_squared = beta ** 2

    result_nom = (beta_squared + 1) * (precision * recall)
    result_denom = (beta_squared * precision + recall + K.epsilon())
    return  result_nom / result_denom

# This code defines a function that calculates the accuracy score for a 
# given set of true labels and predicted labels.

#@tf.function
@keras.saving.register_keras_serializable(name="accuracy_score")
def accuracy_score(y_true, y_pred, epsilon=1e-4):
    # casting the true labels and predicted labels to float32
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(tf.greater(tf.cast(y_pred, tf.float32), tf.constant(0.5)), tf.float32)
    
    # counting the true positives
    tp = tf.reduce_sum(y_true * y_pred, axis=1)
    
    # counting the false positives
    fp = tf.reduce_sum(y_pred, axis=1) - tp
    
    # counting the false negatives
    fn = tf.reduce_sum(y_true, axis=1) - tp
    
    # casting the true labels and predicted labels to boolean
    y_true = tf.cast(y_true, tf.bool)
    y_pred = tf.cast(y_pred, tf.bool)
    
    # counting the true negatives
    tn = tf.reduce_sum(tf.cast(tf.logical_not(y_true), tf.float32) * tf.cast(tf.logical_not(y_pred), tf.float32), 
                       axis=1)
    # calculating the accuracy score
    return (tp + tn) / (tp + tn + fp + fn + epsilon)


# import tensorflow as tf
# from keras import backend as K

# # from keras.saving import saving_utils
# # register_keras_serializable = saving_utils.register_keras_serializable

# from keras.saving import register_keras_serializable

# # Defining a function that calculates the F-beta score for a given set of 
# # #true labels and predicted labels.
# # The function balances precision and recall and it is useful when there is 
# # an imbalance in the number of positive and negative examples in the data.



# @register_keras_serializable()
# def fbeta(y_true, y_pred, threshold_shift=0):
#     beta = 2

#     # Clipping y_pred between 0 and 1
#     y_pred = K.clip(y_pred, 0, 1)

#     # Rounding y_pred to binary values
#     y_pred_bin = K.round(y_pred + threshold_shift)

#     # Counting true positives, false positives, and false negatives
#     tp = K.sum(K.round(y_true * y_pred_bin)) + K.epsilon()
#     fp = K.sum(K.round(K.clip(y_pred_bin - y_true, 0, 1)))
#     fn = K.sum(K.round(K.clip(y_true - y_pred, 0, 1)))

#     # Calculating precision and recall
#     precision = tp / (tp + fp)
#     recall = tp / (tp + fn)

#     beta_squared = beta ** 2

#     result_nom = (beta_squared + 1) * (precision * recall)
#     result_denom = (beta_squared * precision + recall + K.epsilon())
#     return  result_nom / result_denom

# # This code defines a function that calculates the accuracy score for a 
# # given set of true labels and predicted labels.
# @register_keras_serializable()
# def accuracy_score(y_true, y_pred, epsilon = 1e-4):
#     # casting the true labels and predicted labels to float32
#     y_true = tf.cast(y_true, tf.float32)
#     y_pred = tf.cast(tf.greater(tf.cast(y_pred, tf.float32), tf.constant(0.5)), tf.float32)
    
#     # counting the true positives
#     tp = tf.reduce_sum(y_true * y_pred, axis = 1)
    
#     # counting the false positives
#     fp = tf.reduce_sum(y_pred, axis = 1) - tp
    
#     # counting the false negatives
#     fn = tf.reduce_sum(y_true, axis = 1) - tp
    
#     # casting the true labels and predicted labels to boolean
#     y_true = tf.cast(y_true, tf.bool)
#     y_pred = tf.cast(y_pred, tf.bool)
    
#     # counting the true negatives
#     tn = tf.reduce_sum(tf.cast(tf.logical_not(y_true), tf.float32) * tf.cast(tf.logical_not(y_pred), tf.float32), 
#                        axis = 1)
#     #calculating the accuracy score
#     return (tp + tn)/(tp + tn + fp + fn + epsilon)