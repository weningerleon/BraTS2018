import pickle
import numpy as np
from directories import *


def load_normalized_data(ids=0, dir=savedir_preproc_train1):
    list_data = []
    list_info = []

    for s in os.listdir(dir):
        if s.endswith('.npy'):
            list_data.append(s)
        elif s.endswith('.pkl'):
            list_info.append(s)
        else:
            print(s)
            raise FileNotFoundError("wrong filetype in directory")

    if ids==0:
        ids = range(list_info.__len__())

    list_data.sort()
    list_info.sort()

    dataset = []
    for i in ids:
        infofile = list_info[i]
        datafile = list_data[i]

        assert (infofile[:-3]==datafile[:-3]), "data and info does not match!"

        info = pickle.load(open(opj(dir, infofile),"rb"))
        data = np.load(opj(dir, datafile))
        info['data'] = data

        dataset.append(info)

    return dataset
