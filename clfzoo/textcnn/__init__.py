# -*- coding: utf-8 -*-

import os
from .model import TextCNN

from clfzoo.config import ConfigTextCNN
from clfzoo.instance import Instance

clf_model = None
config = None
instance = None


def model(cfg=None, training=False):
    global clf_model, config, instance

    if cfg is None:
        cfg = ConfigTextCNN()

    if not isinstance(cfg, ConfigTextCNN):
        raise ValueError('The config of TextCNN should inherited from ConfigTextCNN')

    config = cfg
    instance = Instance(config, training=training)

    cfg.logger.info("Init model...")
    clf_model = TextCNN(instance.vocab, cfg)

    if not training:
        cfg.logger.info("Restore model...")
        clf_model.restore()


def train():
    global clf_model, config, instance

    config.logger.info("Convert text to id")
    instance.dataloader.convert_to_ids(instance.vocab)

    config.logger.info("Start to train")
    pad_id = instance.vocab.get_word_idx(instance.vocab.pad_token)

    clf_model.train(instance.dataloader, pad_id)


def predict(datas):
    global clf_model, config, instance

    instance.set_data(datas)
    pad_id = instance.vocab.get_word_idx(instance.vocab.pad_token)

    pred_data = instance.dataloader.next_batch('predict', config.batch_size, pad_id, shuffle=False)
    preds, _ = clf_model.evaluate(pred_data)

    return preds


def test(datas, labels):
    global clf_model, config, instance

    instance.set_data(datas, labels)
    pad_id = instance.vocab.get_word_idx(instance.vocab.pad_token)

    pred_data = instance.dataloader.next_batch('predict', config.batch_size, pad_id, shuffle=False)
    preds, metrics = clf_model.evaluate(pred_data)

    return preds, metrics
