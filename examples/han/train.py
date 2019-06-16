# -*- coding: utf-8 -*-

import sys
sys.path.append('../..')

import clfzoo
import clfzoo.han as clf
from clfzoo.config import ConfigHAN

class Config(ConfigHAN):
    def __init__(self):
        super(Config, self).__init__()
    
    gpu = 0

    log_per_batch = 2
    batch_size = 32
    epochs = 50
    lr_rate = 1e-3

    max_sent_num = 25
    max_sent_len = 60

    train_file = '../data/smp2019_ecdt/SMP2019.train'
    dev_file = '../data/smp2019_ecdt/SMP2019.test'

clf.model(Config(), training=True)
clf.train()
