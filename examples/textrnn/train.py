# -*- coding: utf-8 -*-

import sys

sys.path.append('../..')

import clfzoo
import clfzoo.textrnn as clf
from clfzoo.config import ConfigTextRNN


class Config(ConfigTextRNN):
    def __init__(self):
        super(Config, self).__init__()

    gpu = 0
    log_per_batch = 3

    epochs = 50
    batch_size = 64
    max_sent_num = 1
    max_sent_len = 60

    train_file = '../data/smp2019_ecdt/SMP2019.train'
    dev_file = '../data/smp2019_ecdt/SMP2019.test'


clf.model(Config(), training=True)
clf.train()
