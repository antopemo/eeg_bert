from tensorflow.keras.layers import Layer
import tensorflow as tf
import numpy as np


class SplitterLayer(Layer):

    def call(self, inputs):
        transposed = tf.transpose(inputs)
        zone_1 = tf.transpose(tf.gather(transposed,
                                        np.array([1, 3, 4, 7, 8, 9, 10, 11, 16, 17, 18, 19, 20, 21]) - 1))
        zone_2 = tf.transpose(tf.gather(transposed,
                                        np.array([17, 18, 19, 20, 21, 27, 28, 29, 30, 31, 36, 37, 38, 39, 40, 41]) - 1))
        zone_3 = tf.transpose(tf.gather(transposed,
                                        np.array([37, 38, 39, 40, 41, 47, 48, 49, 50, 51]) - 1))
        zone_4 = tf.transpose(tf.gather(transposed,
                                        np.array([56, 57, 58, 59, 62, 63]) - 1))
        zone_5 = tf.transpose(tf.gather(transposed,
                                        np.array([59, 60, 61, 63, 64]) - 1))
        zone_6 = tf.transpose(tf.gather(transposed,
                                        np.array([41, 42, 43, 44, 45, 46, 51, 52, 53, 54, 55, 56]) - 1))
        zone_7 = tf.transpose(tf.gather(transposed,
                                        np.array([21, 22, 23, 24, 25, 31, 32, 33, 34, 35, 41, 42, 43, 44, 45, 46]) - 1))
        zone_8 = tf.transpose(tf.gather(transposed,
                                        np.array([2, 5, 6, 11, 12, 13, 14, 15, 21, 22, 23, 24, 25, 26]) - 1))

        return zone_1, zone_2, zone_3, zone_4, zone_5, zone_6, zone_7, zone_8
