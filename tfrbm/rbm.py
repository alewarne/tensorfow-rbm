from __future__ import print_function

import tensorflow as tf
import numpy as np
import sys
from .util import tf_xavier_init
from sklearn.neural_network import BernoulliRBM


class RBM:
    def __init__(self,
                 n_visible,
                 n_hidden,
                 learning_rate=0.01,
                 momentum=0.95,
                 xavier_const=1.0,
                 err_function='mse',
                 use_tqdm=False,
                 # DEPRECATED:
                 tqdm=None):
        if not 0.0 <= momentum <= 1.0:
            raise ValueError('momentum should be in range [0, 1]')

        if err_function not in {'mse', 'cosine'}:
            raise ValueError('err_function should be either \'mse\' or \'cosine\'')

        self._use_tqdm = use_tqdm
        self._tqdm = None

        if use_tqdm or tqdm is not None:
            from tqdm import tqdm
            self._tqdm = tqdm

        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        self.momentum = momentum

        # self.x = tf.placeholder(tf.float32, [None, self.n_visible])
        self.x = tf.sparse.placeholder(tf.float32, [None, self.n_visible])
        self.y = tf.placeholder(tf.float32, [None, self.n_hidden])
        self.batch_size = tf.placeholder(tf.float32)

        self.w = tf.Variable(tf_xavier_init(self.n_visible, self.n_hidden, const=xavier_const), dtype=tf.float32)
        self.visible_bias = tf.Variable(tf.zeros([self.n_visible]), dtype=tf.float32)
        self.hidden_bias = tf.Variable(tf.zeros([self.n_hidden]), dtype=tf.float32)

        self.delta_w = tf.Variable(tf.zeros([self.n_visible, self.n_hidden]), dtype=tf.float32)
        self.delta_visible_bias = tf.Variable(tf.zeros([self.n_visible]), dtype=tf.float32)
        self.delta_hidden_bias = tf.Variable(tf.zeros([self.n_hidden]), dtype=tf.float32)

        self.update_weights = None
        self.update_deltas = None
        self.compute_hidden = None
        self.compute_visible = None
        self.compute_visible_from_hidden = None
        self.hidden_recon_p = None

        self._initialize_vars()

        assert self.update_weights is not None
        assert self.update_deltas is not None
        assert self.compute_hidden is not None
        assert self.compute_visible is not None
        assert self.compute_visible_from_hidden is not None
        assert self.hidden_recon_p is not None

        if err_function == 'cosine':
            x1_norm = tf.nn.l2_normalize(self.x, 1)
            x2_norm = tf.nn.l2_normalize(self.compute_visible, 1)
            cos_val = tf.reduce_mean(tf.reduce_sum(tf.mul(x1_norm, x2_norm), 1))
            self.compute_err = tf.acos(cos_val) / tf.constant(np.pi)
        else:
            # self.compute_err = tf.reduce_mean(tf.square(self.x - self.compute_visible))
            self.compute_err = tf.reduce_mean(tf.square(tf.sparse_add(self.x, - self.compute_visible)))

        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    def _initialize_vars(self):
        pass

    def get_err(self, batch_x):
        coo = batch_x.tocoo()
        indices = np.mat([coo.row, coo.col]).transpose()
        return self.sess.run(self.compute_err, feed_dict={self.x: (indices, coo.data, coo.shape)})
        # return self.sess.run(self.compute_err, feed_dict={self.x: batch_x})

    def get_free_energy(self, batch_x):
        coo = batch_x.tocoo()
        indices = np.mat([coo.row, coo.col]).transpose()
        return self.sess.run(self.free_energy, feed_dict={self.x: (indices, coo.data, coo.shape),
                                                          self.batch_size:batch_x.shape[0]})

    def transform(self, batch_x):
        coo = batch_x.tocoo()
        indices = np.mat([coo.row, coo.col]).transpose()
        return self.sess.run(self.compute_hidden, feed_dict={self.x: (indices, coo.data, coo.shape)})
        # return self.sess.run(self.compute_hidden, feed_dict={self.x: batch_x})

    def transform_inv(self, batch_y):
        return self.sess.run(self.compute_visible_from_hidden, feed_dict={self.y: batch_y})

    def reconstruct(self, batch_x):
        return self.sess.run(self.compute_visible, feed_dict={self.x: batch_x})

    def partial_fit(self, batch_x, batch_x_tilde):
        coo = batch_x.tocoo()
        indices = np.mat([coo.row, coo.col]).transpose()
        ret = self.sess.run([self.update_weights + self.update_deltas + [self.hidden_recon_p]], feed_dict={
                                                                                self.x: (indices, coo.data, coo.shape),
                                                                                self.y: batch_x_tilde,
                                                                                self.batch_size: batch_x.shape[0]
                                                                                                        })
        # ret = self.sess.run([self.update_weights + self.update_deltas + [self.hidden_recon_p]],
        #                     feed_dict={self.x: batch_x, self.y: batch_x_tilde, self.batch_size: batch_x.shape[0]})
        # return the hidden reconstruction for the persitent contrastive divergence algorithm
        return ret[0][-1]

    def get_hidden_recon_p(self):
        return self.sess.run(self.hidden_recon_p)

    def get_likelihood(self, batch_x):
        W, vb, hb = self.get_weights()
        rbm = BernoulliRBM(n_components=batch_x.shape[1])
        rbm.components_ = W.T
        rbm.intercept_hidden_ = hb
        rbm.intercept_visible_ = vb
        avg_likelihood = 0
        N = 10
        for _ in range(N):
            avg_likelihood += rbm.score_samples(batch_x).mean()/N
        return avg_likelihood

    def fit(self,
            data_x,
            n_epoches=10,
            batch_size=10,
            shuffle=True,
            verbose=True):
        assert n_epoches > 0

        n_data = data_x.shape[0]
        if batch_size > 0:
            n_batches = n_data // batch_size + (0 if n_data % batch_size == 0 else 1)
        else:
            n_batches = 1

        if shuffle:
            data_x_cpy = data_x.copy()
            inds = np.arange(n_data)
        else:
            data_x_cpy = data_x

        errs = []

        x_tilde = np.zeros((batch_size, self.n_hidden))
        # x_tilde = self.transform(data_x[:batch_size])
        for e in range(n_epoches):
            if verbose and not self._use_tqdm:
                print('Epoch: {:d}'.format(e))

            epoch_errs = np.zeros((n_batches,))
            epoch_errs_ptr = 0

            if shuffle:
                np.random.shuffle(inds)
                data_x_cpy = data_x_cpy[inds]

            r_batches = range(n_batches-1)

            if verbose and self._use_tqdm:
                r_batches = self._tqdm(r_batches, desc='Epoch: {:d}'.format(e), ascii=True, file=sys.stdout)

            for b in r_batches:
                batch_x = data_x_cpy[b * batch_size:(b + 1) * batch_size]
                # if sp.isspmatrix_csr(batch_x):
                #     batch_x = batch_x.toarray()
                x_tilde_new = self.partial_fit(batch_x, x_tilde)
                # persistent contrastive divergence
                x_tilde = x_tilde_new
                batch_err = self.get_err(batch_x)
                epoch_errs[epoch_errs_ptr] = batch_err
                epoch_errs_ptr += 1

            if verbose:
                err_mean = epoch_errs.mean()
                if self._use_tqdm:
                    self._tqdm.write('Train error: {:.8f}'.format(err_mean))
                    self.epoch_evaluation(data_x_cpy)
                    self._tqdm.write('')
                else:
                    print('Train error: {:.8f}'.format(err_mean))
                    print('')
                sys.stdout.flush()

            errs = np.hstack([errs, epoch_errs])

        return errs

    def epoch_evaluation(self, data_x_cpy):
        W, h_v, h_h = self.get_weights()
        transformed = self.transform(data_x_cpy)
        self._tqdm.write('Likelihood: {:.4f}'.format(self.get_likelihood(data_x_cpy)))
        e = self.get_free_energy(data_x_cpy)
        self._tqdm.write('Free energy: {:.4f}'.format(e))
        self._tqdm.write('Avg activation: {:.4f}, Avg max activation: {:.4f}, Avg min activation: {:.4f}'.format(
                                                                                                transformed.mean(),
                                                                                np.max(transformed, axis=1).mean(),
                                                                                np.min(transformed, axis=1).mean()
                                                                                                                    )
                        )
        self._tqdm.write('Avg hidden bias: {:.4f}, max hidden bias: {:.4f}, min hidden bias: {:.4f}'.format(h_h.mean(),
                                                                                                    np.max(h_h),
                                                                                                    np.min(h_h)
                                                                                                    )
                         )
        self._tqdm.write('Avg visible bias: {:.4f}, max visible bias: {:.4f}, min visible bias: {:.4f}'.format(h_v.mean(),
                                                                                                        np.max(h_v),
                                                                                                        np.min(h_v)
                                                                                                        )
                         )

    def get_weights(self):
        return self.sess.run(self.w),\
            self.sess.run(self.visible_bias),\
            self.sess.run(self.hidden_bias)

    def save_weights(self, filename, name):
        saver = tf.train.Saver({name + '_w': self.w,
                                name + '_v': self.visible_bias,
                                name + '_h': self.hidden_bias})
        return saver.save(self.sess, filename)

    def set_weights(self, w, visible_bias, hidden_bias):
        self.sess.run(self.w.assign(w))
        self.sess.run(self.visible_bias.assign(visible_bias))
        self.sess.run(self.hidden_bias.assign(hidden_bias))

    def load_weights(self, filename, name):
        saver = tf.train.Saver({name + '_w': self.w,
                                name + '_v': self.visible_bias,
                                name + '_h': self.hidden_bias})
        saver.restore(self.sess, filename)
