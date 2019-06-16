# -*- coding: utf-8 -*-

import sys
sys.path.append('../..')

import clfzoo
import clfzoo.textcnn as clf
from clfzoo.config import ConfigTextCNN

class Config(ConfigTextCNN):
    def __init__(self):
        super(Config, self).__init__()

    batch_size = 8

    max_sent_num = 1
    max_sent_len = 60
    max_char_len = 10

    train_file = '../data/smp2019_ecdt/SMP2019.train'
    dev_file = '../data/smp2019_ecdt/SMP2019.test'


clf.model(Config())

datas = ['打开QQ游戏', '红烧肉怎么做']
labels = ['app', 'cookbook']
preds, metrics = clf.test(datas, labels)
print(preds)
print(metrics)
