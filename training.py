from tfrbm import BBRBM
import scipy.sparse as sp
import numpy as np
import pickle as pkl
import argparse
from sklearn.neural_network import BernoulliRBM


def get_data(data_path, label_path):
    np.random.seed(42)
    data = sp.load_npz(data_path)[:10000]
    # labels = np.argmax(np.load(label_path), axis=1)
    # data_1 = data[np.where(labels == 1)]
    # random_indices = np.random.choice(np.where(labels == 0)[0], data_1.shape[0], replace=False)
    # data = sp.vstack([data_1, data[random_indices]])
    return data


def get_likelihood(data, W, vb, hb):
    rbm = BernoulliRBM(n_components=W.shape[1])
    rbm.components_ = W.T
    rbm.intercept_hidden_ = hb
    rbm.intercept_visible_ = vb
    return rbm.score_samples(data).mean()


def train_new_fun(args):
    data = get_data(args.data_path, args.label_path)
    filename = 'weights_{}_{}_{}.pkl'.format(args.n_hidden, args.learning_rate, args.epochs)
    bbrbm = BBRBM(n_visible=data.shape[1], n_hidden=args.n_hidden, learning_rate=args.learning_rate,
                  momentum=args.momentum, use_tqdm=True)
    bbrbm.fit(data, n_epoches=args.epochs, batch_size=args.batch_size)
    W, vb, hb = bbrbm.get_weights()
    print(get_likelihood(data, W, vb, hb))
    pkl.dump([W, vb, hb], open(filename, 'wb'))


def train_on_fun(args):
    data = get_data(args.data_path, args.label_path)
    filename = 'trained_on_{}_'.format(args.epochs) + args.param_path
    params = pkl.load(open(args.param_path, 'rb'))
    W, vb, hb = params[0], params[1], params[2]
    bbrbm = BBRBM(n_visible=data.shape[1], n_hidden=W.shape[1], learning_rate=args.learning_rate,
                  momentum=args.momentum, use_tqdm=True)
    bbrbm.set_weights(W, vb, hb)
    bbrbm.fit(data, n_epoches=args.epochs, batch_size=args.batch_size)
    W, vb, hb = bbrbm.get_weights()
    print(get_likelihood(data, W, vb, hb))
    pkl.dump([W, vb, hb], open(filename, 'wb'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train rbms.")
    # sub commands
    subparsers = parser.add_subparsers(dest="command")
    # calculate pcr removal
    train_new = subparsers.add_parser("train_new", help="Calc prototype for data.")
    train_new.add_argument("data_path", type=str, help="Path to data to analyze.")
    train_new.add_argument("label_path", type=str, help="Path to labels to analyze.")
    train_new.add_argument("n_hidden", type=int, help="number of hidden units.")
    train_new.add_argument("learning_rate", type=float, help="learning rate.")
    train_new.add_argument("momentum", type=float, help="Momentum param.")
    train_new.add_argument("epochs", type=int, help="epochs.")
    train_new.add_argument("batch_size", type=int, help="batch_size.")

    train_on = subparsers.add_parser("train_on", help="Calc prototype for data.")
    train_on.add_argument("data_path", type=str, help="Path to data to analyze.")
    train_on.add_argument("label_path", type=str, help="Path to labels to analyze.")
    train_on.add_argument("param_path", type=str, help="Path to parameter file.")
    train_on.add_argument("learning_rate", type=float, help="learning rate.")
    train_on.add_argument("momentum", type=float, help="Momentum param.")
    train_on.add_argument("epochs", type=int, help="epochs.")
    train_on.add_argument("batch_size", type=int, help="batch_size.")
    args = parser.parse_args()
    for k, v in vars(args).items():
        print('{} = {}'.format(k, v))
    if args.command == "train_new":
        train_new_fun(args)
    elif args.command == "train_on":
        train_on_fun(args)
