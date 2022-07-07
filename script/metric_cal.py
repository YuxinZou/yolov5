gt_file = '../txt/gt.txt'

sdk_file = '../txt/sdk.txt'
bytetrack_file = '../txt/bytetrack.txt'
strongsort_file = '../txt/strongsort.txt'


def parse(txt_file):
    res = {}
    data = open(txt_file, 'r', encoding='UTF-8').readlines()
    for d in data:
        d = d.strip().split(' ')
        res[d[0].split('/')[-1]] = list(map(int, d[1:]))
    return res


if __name__ == '__main__':
    res = parse(sdk_file)
    print(res)
