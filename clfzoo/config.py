# -*- coding: utf-8 -*- 

from __future__ import absolute_import, unicode_literals

# -------------------------------------------#
# author: sean lee                           #
# email: xmlee97@gmail.com                   #
# --------------------------------------------#

import sys
import time

if sys.version_info[0] == 2:
    reload(sys)
    sys.setdefaultencoding('utf8')

import os
import logging

ROOT = os.sep.join(os.path.dirname(os.path.abspath(__file__)).split(os.sep)[:-1])


def get_logger(filename):
    """Return a logger instance that writes in filename
    Args:
        filename: (string) path to log.txt
    Returns:
        logger: (instance of logger)
    """
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(format='%(message)s', level=logging.DEBUG)
    handler = logging.FileHandler(filename)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    logging.getLogger().addHandler(handler)

    return logger


class BaseConfig(object):

    def __init__(self):
        # makedir 
        if not os.path.exists(self.vocab_dir):
            os.makedirs(self.vocab_dir)

        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu)  # setting gpu

    def init_dir(model_name):
        ts = str(int(time.time()))
        graph_dir = os.path.join(ROOT, "models/{}/{}/graph/".format(model_name, ts))
        model_dir = os.path.join(ROOT, "models/{}/{}/ckpt/".format(model_name, ts))
        vocab_dir = os.path.join(ROOT, "models/{}/{}/vocab/".format(model_name, ts))

        logger_file_path = os.path.join(ROOT, "models/{}/{}/log.txt".format(model_name, ts))
        # makedirs
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        if not os.path.exists(graph_dir):
            os.makedirs(graph_dir)
        if not os.path.exists(vocab_dir):
            os.makedirs(vocab_dir)
        if not os.path.exists(logger_file_path):
            with open(logger_file_path, 'w', encoding='utf-8') as f:
                f.close()
        logger = get_logger(logger_file_path)

        return graph_dir, model_dir, vocab_dir, logger

    splitter = '\t'  # the splitter between label and sentence in train / dev corpus

    train_file = None
    dev_file = None
    test_file = None

    max_sent_num = 1  # choose {max_sent_num} sentences. for model `HAN` the value should greater than 1 others equal 1
    max_sent_len = 100  # the max length of sentence
    max_char_len = 8  # the max length of chars

    min_word_freq = 2  # filter the words whose frequence lower than {min_word_freq}

    log_per_batch = 100  # save training log each {log_per_batch} batches

    word_embed_dim = 100  # word embedding size
    char_embed_dim = 64  # char embedding size

    use_pretrained_embedding = False  # whether to use pretrained word embedding
    pretrained_embedding_file = '/path/to/embedding_path'
    embedding_kernel = 'kv'  # kv | gensim, `kv` means it is a dict {word: embedding}, gensim means it is a gensim model

    gpu = 0  # choose gpu device, {gpu} < 0 means default use cpu


class ConfigTextCNN(BaseConfig):
    def __init__(self):
        super(ConfigTextCNN, self).__init__()

    model_name = 'textcnn'
    graph_dir, model_dir, vocab_dir, logger = BaseConfig.init_dir(model_name)

    # hyparams
    lr_rate = 1e-3  # learning rate
    lr_decay = 0.9  # learning rete decay
    dropout = 0.5  # dropout rate, Attention! it is not keep_prob
    optimizer = 'adam'
    clipper = 5  # grad clipper, less than 0 means don`t applied
    batch_size = 16
    epochs = 10
    early_stop = 10  # early stop
    eval_metric = 'acc'  # p | r | f1 | acc

    optimizer = 'adam'  # adam | sgd | adagrad | adadelta
    loss_type = 'cross_entropy'  # cross_entropy | focal_loss

    kernel_sizes = [3]  # you can set multi-channels
    filter_size = 128


class ConfigTextRNN(BaseConfig):
    def __init__(self):
        super(ConfigTextRNN, self).__init__()

    model_name = 'textrnn'
    graph_dir, model_dir, vocab_dir, logger = BaseConfig.init_dir(model_name)

    # hyparams
    lr_rate = 1e-3
    lr_decay = 0.9
    dropout = 0.5
    optimizer = 'adam'
    clipper = -1
    batch_size = 16
    epochs = 10
    early_stop = 10
    eval_metric = 'acc'  # p | r | f1 | acc

    hidden_dim = 128

    optimizer = 'adam'  # adam | sgd | adagrad | adadelta
    loss_type = 'cross_entropy'  # cross_entropy | focal_loss


class ConfigTransformer(BaseConfig):
    def __init__(self):
        super(ConfigTransformer, self).__init__()

    model_name = 'transformer'
    graph_dir, model_dir, vocab_dir, logger = BaseConfig.init_dir(model_name)

    # hyparams
    lr_rate = 1e-3
    lr_decay = 0.9
    dropout = 0.5
    optimizer = 'adam'
    clipper = -1
    batch_size = 16
    epochs = 20
    early_stop = 10
    eval_metric = 'acc'  # p | r | f1 | acc

    num_heads = 4
    num_blocks = 2
    num_units = 256

    optimizer = 'adam'  # adam | sgd | adagrad | adadelta
    loss_type = 'focal_loss'  # cross_entropy | focal_loss


class ConfigDPCNN(BaseConfig):
    def __init__(self):
        super(ConfigDPCNN, self).__init__()

    model_name = 'dpcnn'
    graph_dir, model_dir, vocab_dir, logger = BaseConfig.init_dir(model_name)

    # hyparams
    lr_rate = 1e-3
    lr_decay = 0.9
    dropout = 0.5
    optimizer = 'adam'
    clipper = -1
    batch_size = 16
    epochs = 10
    early_stop = 10
    eval_metric = 'acc'  # p | r | f1 | acc

    optimizer = 'adam'  # adam | sgd | adagrad | adadelta
    loss_type = 'cross_entropy'  # cross_entropy | focal_loss

    num_blocks = 2
    filter_size = 250


class ConfigHAN(BaseConfig):
    def __init__(self):
        super(ConfigHAN, self).__init__()

    model_name = 'han'
    graph_dir, model_dir, vocab_dir, logger = BaseConfig.init_dir(model_name)

    # hyparams
    lr_rate = 1e-3
    lr_decay = 0.9
    dropout = 0.5
    optimizer = 'adam'
    clipper = -1
    batch_size = 16
    epochs = 10
    early_stop = 10
    eval_metric = 'acc'  # p | r | f1 | acc

    optimizer = 'adam'  # adam | sgd | adagrad | adadelta
    loss_type = 'cross_entropy'  # cross_entropy | focal_loss

    max_sent_num = 2
    num_heads = 1
    hidden_dim = 128


"""Configure for RCNN
"""


class ConfigRCNN(BaseConfig):
    def __init__(self):
        super(ConfigRCNN, self).__init__()

    model_name = 'rcnn'
    graph_dir, model_dir, vocab_dir, logger = BaseConfig.init_dir(model_name)

    # hyparams
    lr_rate = 1e-3  # learning rate
    lr_decay = 0.9  # learning rete decay
    dropout = 0.5  # dropout rate, Attention! it is not keep_prob
    optimizer = 'adam'
    clipper = -1  # grad clipper, less than 0 means don`t applied
    batch_size = 16
    epochs = 10
    early_stop = 10  # early stop
    eval_metric = 'acc'  # p | r | f1 | acc

    optimizer = 'adam'  # adam | sgd | adagrad | adadelta
    loss_type = 'cross_entropy'  # cross_entropy | focal_loss

    # for cnn
    kernel_sizes = [3]  # you can set multi-channels
    filter_size = 128

    # for rnn
    hidden_dim = 128
