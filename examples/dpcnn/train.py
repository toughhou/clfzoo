# -*- coding: utf-8 -*-

import sys
sys.path.append('../..')

import clfzoo
import clfzoo.dpcnn as clf
from clfzoo.config import ConfigDPCNN

class Config(ConfigDPCNN):
    def __init__(self):
        super(Config, self).__init__()

    gpu = 0

    log_per_batch = 2
    epochs = 20
    batch_size = 32

    max_sent_num = 1
    max_sent_len = 60
    max_char_len = 10

    train_file = '../data/smp2019_ecdt/SMP2019.train'
    dev_file = '../data/smp2019_ecdt/SMP2019.test'


clf.model(Config(), training=True)
clf.train()
