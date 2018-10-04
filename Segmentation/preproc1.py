from dipy.io.image import load_nifti
import numpy as np
import pickle
from directories import *

import multiprocessing as mp
from argparse import ArgumentParser


def normalize_deprecated(image, mask=None):
    if mask is None:
        mask = image != image[0,0,0]
    lower_clip = np.percentile(image[mask != 0].ravel(), 2.5)
    upper_clip = np.percentile(image[mask != 0].ravel(), 97.5)
    image[(image < lower_clip) & (mask !=0)] = lower_clip
    image[(image > upper_clip) & (mask !=0)] = upper_clip

    mean = image[mask].mean()
    std = image[mask].std()

    image = image.astype(dtype=np.float32)

    image[mask] = (image[mask] - mean) / std
    return image


def normalize(image, mask=None):
    if mask is None:
        mask = image != image[0,0,0]

    image = image.astype(dtype=np.float32)
    image[mask] = (image[mask] - image[mask].mean()) / image[mask].std()
    return image


def processlist_train(list_dirs, savedir=savedir_preproc_train1):
    for idx, (dir, subject) in enumerate(list_dirs):
        #%% Load data
        t1, affine = load_nifti(opj(dir, subject + '_t1.nii.gz'))
        t2, a_t2 = load_nifti(opj(dir, subject + '_t2.nii.gz'))
        t1ce, a_t1ce = load_nifti(opj(dir, subject + '_t1ce.nii.gz'))
        flair, a_flair = load_nifti(opj(dir, subject + '_flair.nii.gz'))
        gt, a_gt = load_nifti(opj(dir, subject + '_seg.nii.gz'))

        assert np.array_equal(affine, a_t2), 'affines do not match'
        assert np.array_equal(affine, a_t1ce), 'affines do not match'
        assert np.array_equal(affine, a_flair), 'affines do not match'
        assert np.array_equal(affine, a_gt), 'affines do not match'

        assert np.array_equal(t1.shape, t2.shape), 'shapes do not match'
        assert np.array_equal(t1.shape, t1ce.shape), 'shapes do not match'
        assert np.array_equal(t1.shape, flair.shape), 'shapes do not match'
        assert np.array_equal(t1.shape, gt.shape), 'shapes do not match'

        assert t1[0, 0, 0] == 0, 'non-zero background?!'
        assert t2[0, 0, 0] == 0, 'non-zero background?!'
        assert t1ce[0, 0, 0] == 0, 'non-zero background?!'
        assert flair[0, 0, 0] == 0, 'non-zero background?!'
        assert gt[0, 0, 0] == 0, 'non-zero background?!'

        t1ce_sub_t1 = t1ce - t1

        #%% brain mask
        mask = (t1 != 0) | (t1ce != 0) | (t2 != 0) | (flair != 0) | (gt != 0)

        #%% Extract Brain BBox
        brain_voxels = np.where(mask != 0)
        minZidx = int(np.min(brain_voxels[0]))
        maxZidx = int(np.max(brain_voxels[0]))
        minXidx = int(np.min(brain_voxels[1]))
        maxXidx = int(np.max(brain_voxels[1]))
        minYidx = int(np.min(brain_voxels[2]))
        maxYidx = int(np.max(brain_voxels[2]))

        tumor_voxels = np.where(gt != 0)
        minZidx_tumor = int(np.min(tumor_voxels[0]))
        maxZidx_tumor = int(np.max(tumor_voxels[0]))
        minXidx_tumor = int(np.min(tumor_voxels[1]))
        maxXidx_tumor = int(np.max(tumor_voxels[1]))
        minYidx_tumor = int(np.min(tumor_voxels[2]))
        maxYidx_tumor = int(np.max(tumor_voxels[2]))

        t1 = normalize(t1)
        t2 = normalize(t2)
        t1ce = normalize(t1ce)
        t1sub = normalize(t1ce_sub_t1)
        flair = normalize(flair)
        gt[gt == 4] = 3

        t1_boxed = t1[minZidx:maxZidx,minXidx:maxXidx,minYidx:maxYidx]
        t2_boxed = t2[minZidx:maxZidx,minXidx:maxXidx,minYidx:maxYidx]
        t1ce_boxed = t1ce[minZidx:maxZidx,minXidx:maxXidx,minYidx:maxYidx]
        t1sub_boxed = t1sub[minZidx:maxZidx,minXidx:maxXidx,minYidx:maxYidx]
        flair_boxed = flair[minZidx:maxZidx,minXidx:maxXidx,minYidx:maxYidx]
        gt_boxed = gt[minZidx:maxZidx,minXidx:maxXidx,minYidx:maxYidx]

        all_data = np.zeros([6] + list(t1_boxed.shape), dtype=np.float32)
        all_data[0] = t1_boxed
        all_data[1] = t1ce_boxed
        all_data[2] = t1sub_boxed
        all_data[3] = t2_boxed
        all_data[4] = flair_boxed
        all_data[5] = gt_boxed

        np.save(opj(savedir, subject + ".npy"), all_data)

        bbox_brain = [minZidx, maxZidx, minXidx, maxXidx, minYidx, maxYidx]
        bbox_tumor = [minZidx_tumor, maxZidx_tumor, minXidx_tumor, maxXidx_tumor, minYidx_tumor, maxYidx_tumor]

        info = {'original_shape': t1.shape, 'bbox_brain': bbox_brain, 'bbox_tumor': bbox_tumor, 'name': subject, 'affine': affine}
        pickle.dump(info, open(opj(savedir, subject + ".pkl"), "wb"))

        print("Processed: " + subject)


def processlist_validate(list_dirs, savedir=savedir_preproc_val1):
    for idx, (dir, subject) in enumerate(list_dirs):
        #%% Load data
        t1, affine = load_nifti(opj(dir, subject + '_t1.nii.gz'))
        t2, a_t2 = load_nifti(opj(dir, subject + '_t2.nii.gz'))
        t1ce, a_t1ce = load_nifti(opj(dir, subject + '_t1ce.nii.gz'))
        flair, a_flair = load_nifti(opj(dir, subject + '_flair.nii.gz'))

        assert np.array_equal(affine, a_t2), 'affines do not match'
        assert np.array_equal(affine, a_t1ce), 'affines do not match'
        assert np.array_equal(affine, a_flair), 'affines do not match'

        assert np.array_equal(t1.shape, t2.shape), 'shapes do not match'
        assert np.array_equal(t1.shape, t1ce.shape), 'shapes do not match'
        assert np.array_equal(t1.shape, flair.shape), 'shapes do not match'

        assert t1[0, 0, 0] == 0, 'non-zero background?!'
        assert t2[0, 0, 0] == 0, 'non-zero background?!'
        assert t1ce[0, 0, 0] == 0, 'non-zero background?!'
        assert flair[0, 0, 0] == 0, 'non-zero background?!'

        t1ce_sub_t1 = t1ce - t1

        #%% brain mask
        mask = (t1 != 0) | (t1ce != 0) | (t2 != 0) | (flair != 0)

        #%% Extract Brain BBox
        brain_voxels = np.where(mask != 0)
        minZidx = int(np.min(brain_voxels[0]))
        maxZidx = int(np.max(brain_voxels[0]))
        minXidx = int(np.min(brain_voxels[1]))
        maxXidx = int(np.max(brain_voxels[1]))
        minYidx = int(np.min(brain_voxels[2]))
        maxYidx = int(np.max(brain_voxels[2]))

        t1 = normalize(t1)
        t2 = normalize(t2)
        t1ce = normalize(t1ce)
        t1sub = normalize(t1ce_sub_t1)
        flair = normalize(flair)

        t1_boxed = t1[minZidx:maxZidx,minXidx:maxXidx,minYidx:maxYidx]
        t2_boxed = t2[minZidx:maxZidx,minXidx:maxXidx,minYidx:maxYidx]
        t1ce_boxed = t1ce[minZidx:maxZidx,minXidx:maxXidx,minYidx:maxYidx]
        t1sub_boxed = t1sub[minZidx:maxZidx,minXidx:maxXidx,minYidx:maxYidx]
        flair_boxed = flair[minZidx:maxZidx,minXidx:maxXidx,minYidx:maxYidx]

        all_data = np.zeros([6] + list(t1_boxed.shape), dtype=np.float32)
        all_data[0] = t1_boxed
        all_data[1] = t1ce_boxed
        all_data[2] = t1sub_boxed
        all_data[3] = t2_boxed
        all_data[4] = flair_boxed

        np.save(opj(savedir, subject + ".npy"), all_data)

        bbox_brain = [minZidx, maxZidx, minXidx, maxXidx, minYidx, maxYidx]
        bbox_tumor = [0,0,0,0,0,0]
        info = {'original_shape': t1.shape, 'bbox_brain': bbox_brain, 'bbox_tumor': bbox_tumor, 'name': subject, 'affine': affine}
        pickle.dump(info, open(opj(savedir, subject + ".pkl"), "wb"))

        print("Processed: " + subject)


def main(num_processes=4, mode='train'):
    if mode == "train":
        target=processlist_train
        raw_dir = raw_dir_train
        save_dir = savedir_preproc_train1
    elif mode == "validation":
        target = processlist_validate
        raw_dir = raw_dir_validate
        save_dir = savedir_preproc_val1
    elif mode == "test":
        target = processlist_validate
        raw_dir = raw_dir_test
        save_dir = savedir_preproc_test1
    else:
        print("wrong mode!")
        return

    list_dirs = [] # location, name
    for folder in ("HGG", "LGG", 'unknown'):
        dir = opj(raw_dir,folder)
        if not os.path.exists(dir):
            continue
        p = os.listdir(dir)
        for subject in p:
            list_dirs.append((opj(dir,subject), subject))

    list_of_lists = []
    for i in range(num_processes):
        tmp_list=list_dirs[i::num_processes]
        list_of_lists.append(tmp_list)

    processes = [mp.Process(target=target, args=([mylist, save_dir])) for mylist in list_of_lists]

    for p in processes:
        p.start()

    for p in processes:
        p.join()
    print("all finished!")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-n", "--num_procs", dest="num_processes", type=int, default=4)
    parser.add_argument("-m", "--mode", dest="mode", default='train', help='train, val or test')
    args = parser.parse_args()

    main(num_processes=args.num_processes, mode=args.mode)