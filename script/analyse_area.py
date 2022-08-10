import pickle
import matplotlib.pyplot as plt
import numpy as np

path = open('../json/result.pkl', 'rb')
data = pickle.load(path)

x, y = [], []
for fname, d in data.items():
    print(fname)
    id_dict = d['id_dict']
    ratio_dict = d['ratio_dict']

    for k, v in id_dict.items():

        if k != 3:
            continue
        for k_, v_ in v.items():
            if v_[0] < 5 or v_[0] > 15:
                continue
            x.append(v_[0])
            y.append(v_[1])
x = np.array(x)
y = np.array(y)
plt.scatter(x, y, s=1)
plt.show()