import numpy as np
import pickle as pkl
import scipy.sparse as sp
import os
import tensorflow as tf


def tf_xavier_init(fan_in, fan_out, *, const=1.0, dtype=np.float32):
    # k = const * np.sqrt(6.0 / (fan_in + fan_out))
    # return tf.random_uniform((fan_in, fan_out), minval=-k, maxval=k, dtype=dtype)
    return tf.random_normal((fan_in, fan_out), 0.0, 0.01)


def sample_bernoulli(probs):
    return tf.nn.relu(tf.sign(probs - tf.random_uniform(tf.shape(probs))))


def sample_gaussian(x, sigma):
    return x + tf.random_normal(tf.shape(x), mean=0.0, stddev=sigma, dtype=tf.float32)


# takes a filepath and loads the contained data for the data formats .npy, .pkl, .npz
def load_npy_npz_pkl(path_to_data):
    _, filetype = os.path.splitext(path_to_data)
    if filetype == '.npz':
        data = sp.load_npz(path_to_data)
    elif filetype == '.npy':
        data = np.load(path_to_data)
    elif filetype == '.pkl':
        data = pkl.load(open(path_to_data, 'rb'))
    else:
        print('Could not load filepath! Invalid datatype!')
        data = None
    return data
