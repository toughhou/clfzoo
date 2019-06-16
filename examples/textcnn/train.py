# -*- coding: utf-8 -*-

import sys

sys.path.append('../..')

import clfzoo
import clfzoo.textcnn as clf
from clfzoo.config import ConfigTextCNN
import time
import os


class Config(ConfigTextCNN):
    def __init__(self):
        super(Config, self).__init__()

    log_per_batch = 3
    batch_size = 64
    epochs = 50

    # hyparams
    lr_rate = 1e-3  # learning rate
    lr_decay = 0.9  # learning rete decay
    dropout = 0.5  # dropout rate, Attention! it is not keep_prob
    clipper = 5  # grad clipper, less than 0 means don`t applied
    early_stop = 10  # early stop
    eval_metric = 'acc'  # p | r | f1 | acc

    max_sent_num = 1
    max_sent_len = 60
    max_char_len = 10

    train_file = '../data/smp2019_ecdt/SMP2019.train'
    dev_file = '../data/smp2019_ecdt/SMP2019.test'


clf.model(Config(), training=True)
clf.train()
