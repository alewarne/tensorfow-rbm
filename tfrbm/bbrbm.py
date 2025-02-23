import tensorflow as tf
from .rbm import RBM
from .util import sample_bernoulli


class BBRBM(RBM):
    def __init__(self, *args, **kwargs):
        RBM.__init__(self, *args, **kwargs)

    def _initialize_vars(self):
        hidden_p = tf.nn.sigmoid(tf.sparse.sparse_dense_matmul(self.x, self.w) + self.hidden_bias)
        visible_recon_p = sample_bernoulli(tf.nn.sigmoid(tf.matmul(self.y, tf.transpose(self.w)) + self.visible_bias))
        hidden_recon_p = tf.nn.sigmoid(tf.matmul(visible_recon_p, self.w) + self.hidden_bias)

        positive_grad = tf.sparse.sparse_dense_matmul(tf.sparse.transpose(self.x), hidden_p)
        negative_grad = tf.matmul(tf.transpose(visible_recon_p), hidden_recon_p)

        def momentum(x_old, x_new):
            return self.momentum * x_old +\
                   self.learning_rate * x_new * (1 - self.momentum) / tf.to_float(self.batch_size)

        delta_w_new = momentum(self.delta_w, positive_grad - negative_grad)
        delta_visible_bias_new = momentum(self.delta_visible_bias, tf.reduce_sum(tf.sparse.add(self.x, - visible_recon_p), 0))
        delta_hidden_bias_new = momentum(self.delta_hidden_bias, tf.reduce_sum(hidden_p - hidden_recon_p, 0))

        update_delta_w = self.delta_w.assign(delta_w_new)
        update_delta_visible_bias = self.delta_visible_bias.assign(delta_visible_bias_new)
        update_delta_hidden_bias = self.delta_hidden_bias.assign(delta_hidden_bias_new)

        update_w = self.w.assign(self.w + delta_w_new)
        update_visible_bias = self.visible_bias.assign(self.visible_bias + delta_visible_bias_new)
        update_hidden_bias = self.hidden_bias.assign(self.hidden_bias + delta_hidden_bias_new)

        self.update_deltas = [update_delta_w, update_delta_visible_bias, update_delta_hidden_bias]
        self.update_weights = [update_w, update_visible_bias, update_hidden_bias]

        self.hidden_recon_p = sample_bernoulli(hidden_recon_p)
        self.compute_hidden = tf.nn.sigmoid(tf.sparse.sparse_dense_matmul(self.x, self.w) + self.hidden_bias)
        self.compute_visible = tf.nn.sigmoid(tf.matmul(self.compute_hidden, tf.transpose(self.w)) + self.visible_bias)
        self.compute_visible_from_hidden = tf.nn.sigmoid(tf.matmul(self.y, tf.transpose(self.w)) + self.visible_bias)

        # quantities necessary for computing free energy
        inp_t_h = tf.sparse.sparse_dense_matmul(self.x, tf.reshape(self.visible_bias, shape=(tf.shape(self.visible_bias)[0], 1)))
        inp_t_h = tf.reshape(inp_t_h, shape=(tf.shape(inp_t_h)[0],))
        inp_t_w = tf.sparse.sparse_dense_matmul(self.x, self.w) + self.hidden_bias
        # sum_ log(1+exp(W*x+b_h))
        log_sum = tf.reduce_logsumexp([tf.zeros(shape=(self.batch_size, self.n_hidden)), inp_t_w], axis=0)
        log_sum_sum = tf.reduce_sum(log_sum, axis=1)
        self.free_energy = -inp_t_h - log_sum_sum
