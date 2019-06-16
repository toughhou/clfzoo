# -*- coding: utf-8- -*-


def process(fin, fout):
    out_format = '{}\t{}\n'

    with open(fin, 'r') as rf, open(fout, 'w') as wf:
        for line in rf:
            line = line.strip()
            arr = line.split()
            if len(arr) == 0:
                continue

            wf.writelines(out_format.format(arr[0], ' '.join(arr[1:])))


process('TREC.train.txt', 'TREC.train')
process('TREC.test.txt', 'TREC.test')
