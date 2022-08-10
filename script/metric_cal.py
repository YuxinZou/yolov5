import numpy as np

gt_file = '../txt/gt_full_filter32_remove.txt'

bytetrack_file = '../txt/pt.txt'
tensorrt_file = '../txt/tensorrt.txt'
tensorrt_buffer30_file = '../txt/tensorrt_buffer_30.txt'
interval3 = '../txt/track_buffer90_countthres480_filter32_full.txt'
interval2 = '../txt/track_buffer90_countthres360_filter32_full_inter2_new.txt'


def parse(txt_file):
    res = {}
    data = open(txt_file, 'r', encoding='UTF-8').readlines()
    for d in data:
        d = d.strip().split(' ')
        name = d[0].split('/')[-1].split('.')[0]
        print(d)
        res[name] = list(map(int, d[1:]))
    return res


def metric(data1, data2):
    res = {}
    classes = (0, 1, 2, 3)
    for c in classes:
        tp, fp, fn = 0, 0, 0
        for key, gt in data1.items():
            dt = data2[key]
            g = gt[c]
            d = dt[c]
            tp += min(g, d)
            fp += max(0, d - g)
            fn += max(0, g - d)
        res[c] = dict(tp=tp, fp=fp, fn=fn, precision=tp / (tp + fp), recall=tp / (tp + fn), acc=tp / (tp + fp + fn))
    return res


if __name__ == '__main__':
    import numpy as np
    gt = parse(gt_file)
    tensorrt_res = parse(interval3)
    res = metric(gt, tensorrt_res)
    for k,v in res.items():
        print(v)
    print(np.mean([v['acc'] for k,v in res.items()]))
    print(np.mean([v['precision'] for k, v in res.items()]))
    print(np.mean([v['recall'] for k, v in res.items()]))
