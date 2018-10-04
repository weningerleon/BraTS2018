import load_brats
import numpy as np
import pickle
import random
import itertools
import multiprocessing as mp
from argparse import ArgumentParser

from directories import *


def create_input(data, start_idx, end_idx):

    z3 = start_idx[0]
    x3 = start_idx[1]
    y3 = start_idx[2]

    z4 = data.shape[1] - end_idx[0]
    x4 = data.shape[2] - end_idx[1]
    y4 = data.shape[3] - end_idx[2]

    p = abs(min(0,z3,x3,y3,z4,x4,y4))

    data_pad = np.zeros(np.add(data.shape,[0,2*p,2*p,2*p]))
    if p!=0:
        data_pad[:,p:-p,p:-p,p:-p] = data
    else:
        data_pad = data

    z3 = start_idx[0] + p
    x3 = start_idx[1] + p
    y3 = start_idx[2] + p

    z4 = end_idx[0] + p
    x4 = end_idx[1] + p
    y4 = end_idx[2] + p

    output = data_pad[:, z3:z4, x3:x4, y3:y4]

    assert output.shape == (6,124,124, 124), "Wrong size"

    return output


def create_label_input_random(set, n):
    data = set['data']
    bbox_tumor = set['bbox_tumor']
    bbox_brain = set['bbox_brain']

    shape_pred = np.array([36, 36, 36])
    shape_inputs = np.array([124, 124, 124])
    d_shape = (np.divide((shape_inputs - shape_pred), 2)).astype(np.int16)

    t = bbox_tumor
    bbox_start = np.array([t[0], t[2], t[4]])
    bbox_size = np.array([t[1] - t[0] - shape_pred[0], t[3] - t[2] - shape_pred[1], t[5] - t[4] - shape_pred[2]])
    # the prediction will now be 100% in the tumor bounding box
    # add a little bit of padding
    pad = 20
    bbox_start = bbox_start - pad
    bbox_size = bbox_size + 2 * pad

    # absolute position in cropped image
    t2 = bbox_brain
    start_pred = bbox_start - np.array([t2[0], t2[2], t2[4]])

    inputs = []
    labels = []
    for i in range(n):
        # choose random box in bounding box
        if bbox_size[0] > 0:
            i1 = random.randint(0, bbox_size[0])
        else:
            i1 = 0
        if bbox_size[1] > 0:
            i2 = random.randint(0, bbox_size[1])
        else:
            i2 = 0
        if bbox_size[2] > 0:
            i3 = random.randint(0, bbox_size[2])
        else:
            i3 = 0

        z1 = start_pred[0] + i1
        x1 = start_pred[1] + i2
        y1 = start_pred[2] + i3

        z2 = z1 + shape_pred[0]
        x2 = x1 + shape_pred[1]
        y2 = y1 + shape_pred[2]

        z3 = z1 - d_shape[0]
        x3 = x1 - d_shape[0]
        y3 = y1 - d_shape[0]

        z4 = z2 + d_shape[0]
        x4 = x2 + d_shape[0]
        y4 = y2 + d_shape[0]

        input = create_input(data, [z3, x3, y3], [z4, x4, y4])

        label = input[5,44:-44,44:-44,44:-44]
        label = np.expand_dims(label,0)

        input = input[:5, :, :, :]

        input = np.expand_dims(input, axis=0)
        input = np.float32(input)

        label = np.int64(label)
        label = np.eye(4)[label]
        label = np.moveaxis(label, -1, 1)
        label = np.float32(label)

        inputs.append(input)
        labels.append(label)

    return inputs, labels


def create_label_input_complete(set):
    data = set['data']
    bbox_tumor = set['bbox_tumor']
    bbox_brain = set['bbox_brain']

    shape_pred = np.array([36, 36, 36])
    shape_inputs = np.array([124, 124, 124])
    d_shape = (np.divide((shape_inputs - shape_pred), 2)).astype(np.int16)

    t = bbox_tumor
    bbox_start = np.array([t[0], t[2], t[4]])
    bbox_size = np.array([t[1] - t[0] - shape_pred[0], t[3] - t[2] - shape_pred[1], t[5] - t[4] - shape_pred[2]])
    # the prediction will now be 100% in the tumor bounding box
    # add a little bit of padding
    pad = 0
    bbox_start = bbox_start - pad
    bbox_size = bbox_size + 2 * pad

    # absolute position in cropped image
    t2 = bbox_brain
    start_pred = bbox_start - np.array([t2[0], t2[2], t2[4]])

    bbox_size[bbox_size<0]=0

    labels = []
    inputs = []

    dbf = np.zeros(data.shape[1:])

    if not np.array_equal(np.unique(data[5,:,:,:]), [0.,1.,2.,3.]):
        print("NOT ALL CLASSES In %s" %set['name'])
        return inputs, labels

    #for stepsize in [20,18,12,10,8,5]:
    stepsize1 = max(np.int(np.rint((bbox_size[0]+36) / 4. )),1)
    stepsize2 = max(np.int(np.rint((bbox_size[1]+36) / 4. )),1)
    stepsize3 = max(np.int(np.rint((bbox_size[2]+36) / 4. )),1)
    for (i1,i2,i3) in itertools.product(range(0,bbox_size[0]+36,stepsize1), range(0,bbox_size[1]+36,stepsize2), range(0,bbox_size[1]+36,stepsize3)):
        # choose random box in bounding box
        z1 = start_pred[0] + i1
        x1 = start_pred[1] + i2
        y1 = start_pred[2] + i3

        z2 = z1 + shape_pred[0]
        x2 = x1 + shape_pred[1]
        y2 = y1 + shape_pred[2]

        z3 = z1 - d_shape[0]
        x3 = x1 - d_shape[0]
        y3 = y1 - d_shape[0]

        z4 = z2 + d_shape[0]
        x4 = x2 + d_shape[0]
        y4 = y2 + d_shape[0]

        input = create_input(data, [z3, x3, y3], [z4, x4, y4])

        dbf[z3+44:z4-44,x3+44:x4-44,y3+44:y4-44]=1

        label = input[5,44:-44,44:-44,44:-44]

        # Check if data is adequate for training
        uniques, counts = np.unique(label, return_counts=True)
        if not np.array_equal(uniques, [0., 1., 2., 3.]):
            continue
        if not np.all(np.greater(counts, [320, 320, 320, 320])):
            # 1 Patch = 32^3 = 32768 => 0.1% = 32.8
            continue

        input = input[:5, :, :, :]
        input = np.expand_dims(input, axis=0)
        input = np.float32(input)

        label = np.expand_dims(label,0)
        label = np.int32(label)
        label = np.eye(4)[label]
        label = np.moveaxis(label, -1, 1)
        label = np.float32(label)

        inputs.append(input)
        labels.append(label)

    print("%i datasets from %s, stepsize=%i, %i, %i" %(labels.__len__(),set['name'], stepsize1, stepsize2, stepsize3))
    return inputs, labels


def get_trainset(small_training=False):
    input_pickles = []
    label_pickles = []

    for s in os.listdir(savedir_preproc_train2):
        if s.startswith('input'):
            input_pickles.append(s)
        elif s.startswith('label'):
            label_pickles.append(s)
        else:
            print("wrong file!")
            break

    input_pickles.sort()
    label_pickles.sort()

    if small_training==True:
        input_pickles = input_pickles[:2]
        label_pickles = label_pickles[:2]

    inputs = []
    labels = []

    for input, label in zip(input_pickles, label_pickles):
        cur_inputs = pickle.load(open(opj(savedir_preproc_train2, input), "rb"))
        cur_labels = pickle.load(open(opj(savedir_preproc_train2, label), "rb"))
        inputs.extend(cur_inputs)
        labels.extend(cur_labels)

    return inputs, labels


def main_random_sp(train_sets, n=20):
    for train_set in train_sets:
        inputs, labels = create_label_input_random(train_set, n)
        pickle.dump(inputs, open(opj(savedir_preproc_train2, 'inputs_' + train_set['name'] + '.p'), 'wb'))
        pickle.dump(labels, open(opj(savedir_preproc_train2, 'labels_' + train_set['name'] + '.p'), 'wb'))
        print("random processed: " + train_set['name'])

    return 0


def main_patches(train_sets):
    for train_set in train_sets:
        inputs, labels = create_label_input_complete(train_set)
        pickle.dump(inputs, open(opj(savedir_preproc_train2, 'inputs_' + train_set['name'] + '.p'), 'wb'))
        pickle.dump(labels, open(opj(savedir_preproc_train2, 'labels_' + train_set['name'] + '.p'), 'wb'))
        print("Patch-based processed: " + train_set['name'])

    return 0


def main(num_processes=4, mode='random'):
    train_sets = load_brats.load_normalized_data(dir=savedir_preproc_train1)

    list_of_lists = []
    for i in range(num_processes):
        tmp_list=train_sets[i::num_processes]
        list_of_lists.append(tmp_list)

    if mode == 'random':
        target = main_random_sp
        processes = [mp.Process(target=target, args=([mylist])) for mylist in list_of_lists]
    elif mode == 'patches':
        target = main_patches
        processes = [mp.Process(target=target, args=([mylist])) for mylist in list_of_lists]
    else:
        print("Wrong mode!")
        return

    for p in processes:
        p.start()

    for p in processes:
        p.join()
    print("all finished!")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-n", "--num_procs", dest="num_processes", type=int, default=4)
    parser.add_argument("-m", "--mode", dest="mode", default='random', help='random or patches - how the training patches are chosen from the tumor region: '
                                                                           'either random sampling or in a grid fashion')
    args = parser.parse_args()

    main(num_processes=args.num_processes, mode=args.mode)