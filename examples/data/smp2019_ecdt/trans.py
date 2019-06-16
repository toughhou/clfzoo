# -*- coding: utf-8- -*-
import os
import jieba

basedir = os.path.abspath('') + '/'


def etl(fin, fout):
    return ""


def use_jieba_cut(a_sentence, use_stop_words=False):
    if use_stop_words:
        stopword_set = set()
        with open(basedir + './stopwords.txt', 'r', encoding='utf-8') as stopwords:
            for stopword in stopwords:
                stopword_set.add(stopword.strip('\n'))

        seg_ws = list(jieba.cut(a_sentence))

        result = list()
        for word in seg_ws:
            if word not in stopword_set:
                result.append(word)
    else:
        result = list(jieba.cut(a_sentence))

    return ' '.join(result)


def process(fin, fout):
    out_format = '{}\t{}\n'

    with open(fin, 'r', encoding='utf-8') as rf, open(fout, 'w', encoding='utf-8') as wf:
        for line in rf:
            line = line.strip()
            arr = line.split('\t')

            if len(arr) == 0:
                continue

            label = arr[0]
            seg_text = use_jieba_cut(arr[1], use_stop_words=False)

            wf.writelines(out_format.format(label, seg_text))


process('SMP2019.train.txt', 'SMP2019.train')
process('SMP2019.test.txt', 'SMP2019.test')
