# -*- coding: utf-8 -*-

import sys
sys.path.append('../..')

import clfzoo.rcnn as clf
from clfzoo.config import ConfigRCNN

class Config(ConfigRCNN):
    def __init__(self):
        super(Config, self).__init__()

    # gpu = -1
    log_per_batch = 5

    epochs = 20
    batch_size = 8
    max_sent_num = 1
    max_sent_len = 60

    train_file = '../data/smp2019_ecdt/SMP2019.train'
    dev_file = '../data/smp2019_ecdt/SMP2019.test'


clf.model(Config(), training=True)
clf.train()
