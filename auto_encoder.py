'''
Created on 2016/05/14

@author: matsunagi
'''

import sys
import numpy
from argparse import ArgumentParser
from chainer import Chain, Variable, cuda, functions, links, optimizer, optimizers, serializers
from collections import defaultdict
import pickle


def parse_args():
    def_function = "sigmoid"
    def_threshold = 2
    def_network = 0.1
    def_net_size = 100
    def_maxepoch = 500
    def_train = 10000
    def_batch_size = 100
    p = ArgumentParser(
    description='Auto-encoder practice',
    usage=
    '\n  %(prog)s source_file [options]'
    '\n  %(prog)s -h',
    )
    p.add_argument("source", help="source file")
    p.add_argument("--word_threshold", default=def_threshold, type=int, help="minimum occurring number to be added in the vocablary")
    p.add_argument("--net_threshold", default=def_network, type=float, help="loss value to end learning")
    p.add_argument("--pre_max_epoch", type=int, default=def_maxepoch, help="max number to stop iteration in pre-training")
    p.add_argument("--max_epoch", type=int, default=def_maxepoch, help="max number to stop iteration")
    p.add_argument("--function", default=def_function, help="to be appeared...")
    p.add_argument("--network_size", type=int, default=def_net_size, help="size of hidden layer")
    p.add_argument("--train_size", type=int, default=def_train, help="training data size. the rest of source will be test")
    p.add_argument("--load_vocab", action="store_true", default=False, help="restore vocabulary (pickle format)")
    p.add_argument("--save_vocab", action="store_true", default=False, help="save vocabulary (pickle format)")
    p.add_argument("--vocab", help="(to be) restored vocabulary (pickle format)")
    p.add_argument("--mini_batch", type=int, default=def_batch_size, help="minibatch number")
    p.add_argument("--yes_mini_batch", action="store_true", default=False, help="activate minibatch")
    p.add_argument("--no_pretraining", action="store_true", default=False, help="train a model without pretraining (auto-encoder)")
    p.add_argument("--debug", action="store_true", default=False, help="print iroiro na mono")

    args = p.parse_args()
    return args


class AutoEncoder(Chain):
    def __init__(self, input_size, network_size):
        super(AutoEncoder, self).__init__(
                                          hidden_layer=links.Linear(input_size, network_size),
                                          reconstruct_layer=links.Linear(network_size, input_size),
                                          output=links.Linear(network_size, 1),
                                          )

    def __call__(self, input, pretrain=False):
        h = functions.tanh(self.hidden_layer(input))
        if pretrain:
            return functions.tanh(self.reconstruct_layer(h))
        else:
            return functions.tanh(self.output(h))


def make_vocabulary(source_file, threshold=2):
    # input line:
    # label(1or0) sentence
    vocab = defaultdict(int)
    with open(source_file, "r") as source:
        while True:
            line = source.readline().strip()
            if not line:
                break
            for word in line.split()[1:]:
                vocab[word] += 1

    idn = 0
    idvocab = dict(type=int)
    for k, v in sorted(vocab.items(), key=lambda x: x[1], reverse=True):
        if v < threshold:
            break
        idn += 1
        idvocab[k] = idn
    return idvocab


def make_bow_vector_and_label(line, vocab):
    # make one-hot vector ?
    label = int(line.split()[0])
    words = [vocab[x] for x in set(line.split()[1:]) if x in vocab]
    vector = numpy.array([x in words for x in range(len(vocab))], dtype=numpy.int32)
    return (label, vector)


def evaluate_data(testdata, model):
    test_source, test_target = testdata
    model.zerograds()
    test = [1 if x > 0.5 else 0 for x in model(test_source).data]
    correct = [1 for x, y in zip(test, test_target.data) if x == y]
    print("correct: {0}/{1} ({2:.2f}%)".format(len(correct), len(test_target), len(correct)/len(test_target)*100))


def main(args):
    if args.load_vocab:
        with open(args.vocab, "rb") as vocab_file:
            vocab = pickle.load(vocab_file)
    else:
        vocab = make_vocabulary(args.source, args.word_threshold)
        if args.save_vocab:
            with open(args.vocab, "wb") as vocab_file:
                pickle.dump(vocab, vocab_file)
    print("vocabulary done: {0} words".format(len(vocab)))

    with open(args.source, "r") as sourcefile:
        labels, vectors = [], []
        while True:
            line = sourcefile.readline().strip()
            if not line:
                break
            label, vector = make_bow_vector_and_label(line, vocab)
            labels.append([label])
            vectors.append(vector)
    print("vectors done")

    train_source = numpy.array(vectors[:args.train_size], dtype=numpy.float32)
    train_target = numpy.array(labels[:args.train_size], dtype=numpy.float32)
    test_source = Variable(numpy.array(vectors[args.train_size:], dtype=numpy.float32))
    test_target = Variable(numpy.array(labels[args.train_size:], dtype=numpy.float32))

    """
    if args.function == "sigmoid":
        function = functions.sigmoid
    elif args.function == "relu":
        function = functions.relu
    elif args.function == "tanh":
        function = functions.tanh
    """

    input_size = len(vocab.keys())
    model = AutoEncoder(input_size, args.network_size)
    optimizer = optimizers.Adam()
    # optimizer = optimizers.SGD()
    optimizer.setup(model)
    indexes = numpy.random.permutation(input_size)
    batch_size = args.mini_batch

    if not args.no_pretraining:
        print("start pre-training")
        loss_value = 1
        epoch = 0
        while loss_value > args.net_threshold and epoch < args.max_epoch:
            epoch += 1
            if args.yes_mini_batch:
                for i in range(0, input_size, batch_size):
                    mini_source = Variable(train_source[indexes[i:i + batch_size]])
                    model.zerograds()
                    y = model(mini_source, pretrain=True)
                    loss = functions.mean_squared_error(y, mini_source)
                    loss.backward()
                    optimizer.update()
                    loss_value = loss.data
            else:
                mini_source = Variable(train_source)
                model.zerograds()
                y = model(mini_source, pretrain=True)
                loss = functions.mean_squared_error(y, mini_source)
                loss.backward()
                optimizer.update()
                loss_value = loss.data

            if args.debug and epoch % 20 == 0:
                print("epoch {0}: loss={1}".format(epoch, loss_value))
        print("pre-training done ({0} epochs)".format(epoch))

    print("start normal-training")
    loss_value = 1
    epoch = 0

    while loss_value > args.net_threshold and epoch < args.max_epoch:
        epoch += 1
        if args.yes_mini_batch:
            for i in range(0, input_size, batch_size):
                mini_source = Variable(train_source[indexes[i:i + batch_size]])
                mini_target = Variable(train_target[indexes[i:i + batch_size]])
                model.zerograds()
                y = model(mini_source)
                loss = functions.mean_squared_error(y, mini_target)
                loss.backward()
                optimizer.update()
                loss_value = loss.data
        else:
            mini_source = Variable(train_source)
            mini_target = Variable(train_target)
            model.zerograds()
            y = model(mini_source)
            loss = functions.mean_squared_error(y, mini_target)
            loss.backward()
            optimizer.update()
            loss_value = loss.data

        if args.debug and epoch % 20 == 0:
            print("epoch {0}: loss={1}".format(epoch, loss_value))
    print("pre-training done ({0} epochs)".format(epoch))

    if args.debug:
        index_to_vocab = {vocab[x]: x for x in vocab}
        for i, weights in enumerate(model.hidden_layer.W.data):
            print("weights ", i)
            most_weights = [(list(weights).index(x), x) for x in sorted(weights, reverse=True)][:10]
            for w in most_weights:
                # print(w)
                print("{0} ({1:.3f})".format(index_to_vocab[w[0]], w[1]))

    print("start test")
    model.zerograds()
    predicts = model(test_source)
    test = [1 if x > 0.5 else 0 for x in model(test_source).data]
    correct = [1 for x, y in zip(test, test_target.data) if x == y]
    print("correct: {0}/{1} ({2:.2f}%)".format(len(correct), len(test_target), len(correct)/len(test_target)*100))


if __name__ == '__main__':
    args = parse_args()
    main(args)
