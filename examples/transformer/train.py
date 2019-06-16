# -*- coding: utf-8 -*-

import sys

sys.path.append('../..')

import clfzoo
import clfzoo.transformer as clf
from clfzoo.config import ConfigTransformer


class Config(ConfigTransformer):
    def __init__(self):
        super(Config, self).__init__()

    gpu = 0

    log_per_batch = 2
    batch_size = 128

    epochs = 50
    max_sent_num = 1
    max_sent_len = 50

    lr_rate = 0.05
    num_units = 128
    num_heads = 8
    num_blocks = 4

    train_file = '../data/smp2019_ecdt/SMP2019.train'
    dev_file = '../data/smp2019_ecdt/SMP2019.test'


clf.model(Config(), training=True)
clf.train()
