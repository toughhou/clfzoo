import json
import jieba
import os
import numpy as np
import pandas as pd

basedir = os.path.abspath('') + '/'

RAW_TRAIN_DATA_PATH = basedir + 'train.json'
TRAIN_DATA_PATH = basedir + 'SMP2019.train.txt'
TEST_DATA_PATH = basedir + 'SMP2019.test.txt'

RANDOM_SEED = 40
SPLIT_RATIO = 0.3

COLS_NAME = ['domain', 'query']


def get_json_data(path):
    f = open(path, encoding='utf-8')
    jdata_list = json.load(f)
    data_df = pd.DataFrame(jdata_list)

    return data_df


def split_df(df, ratio):
    data = df.values

    np.random.seed(RANDOM_SEED)
    shuffled_indices = np.random.permutation(len(data))
    test_size = int(len(data) * ratio)
    test_indices = shuffled_indices[:test_size]
    train_indices = shuffled_indices[test_size:]

    train_df = pd.DataFrame(data[train_indices], columns=['domain', 'intent', 'slot', 'query'])
    test_df = pd.DataFrame(data[test_indices], columns=['domain', 'intent', 'slot', 'query'])

    return train_df, test_df


def use_jieba_cut(a_sentence):
    stopword_set = set()

    with open(basedir + './stopwords.txt', 'r', encoding='utf-8') as stopwords:
        for stopword in stopwords:
            stopword_set.add(stopword.strip('\n'))

    seg_ws = list(jieba.cut(a_sentence))

    result = list()
    for word in seg_ws:
        if word not in stopword_set:
            result.append(word)
    seg_ws = result

    return ' '.join(seg_ws)


use_jieba_cut('鱼香茄子怎么做#')

data_df = get_json_data(RAW_TRAIN_DATA_PATH)
train_data_df, test_data_df = split_df(data_df, SPLIT_RATIO)

train_df = train_data_df[COLS_NAME]
train_df['domain'] = train_df['domain'].apply(lambda x: '' + x)
# train_df['query'] = train_df['query'].apply(use_jieba_cut)
train_df.head()
train_df.to_csv(TRAIN_DATA_PATH, sep='\t', header=0, index=0, columns=COLS_NAME)

test_df = test_data_df[COLS_NAME]
test_df['domain'] = test_df['domain'].apply(lambda x: '' + x)
# test_df['query'] = test_df['query'].apply(use_jieba_cut)
test_df.head()
test_df.to_csv(TEST_DATA_PATH, sep='\t', header=0, index=0, columns=COLS_NAME)
