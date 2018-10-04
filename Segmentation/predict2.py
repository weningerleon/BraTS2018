import torch
from torch.autograd import Variable
import numpy as np
from dipy.io.image import load_nifti, save_nifti
from argparse import ArgumentParser
from misc import to_original_shape
import itertools
from skimage.measure import label, regionprops

import load_brats
from directories import *
from unet_nopad import Unet3d_nopad


def predict(predict_sets, net, save_location):
    net = net.cuda()
    net.eval()

    for dataset in predict_sets:

        print("Processing " + dataset['name'] + "...")
        stride = 9
        if dataset['bbox_tumor'] == [0,0,0,0,0,0]:
            print("No tumor bbox found, setting whole brain as tumor mask!!!")

            dataset['data'][5] = (dataset['data'][0] != 0)

            brain_voxels = np.where(dataset['data'][0] != 0)
            minZidx = int(np.min(brain_voxels[0]))
            maxZidx = int(np.max(brain_voxels[0]))
            minXidx = int(np.min(brain_voxels[1]))
            maxXidx = int(np.max(brain_voxels[1]))
            minYidx = int(np.min(brain_voxels[2]))
            maxYidx = int(np.max(brain_voxels[2]))
            dataset['bbox_tumor'] = [minZidx, maxZidx, minXidx, maxXidx, minYidx, maxYidx]

            stride = 9

        data = dataset['data']

        output_shape = [36,36,36]
        pad_net_shape = [44,44,44]

        pad = 80 # 36 + 44
        data_pad = np.pad(data, ((0, 0), (pad, pad), (pad, pad), (pad, pad)),
                          mode='constant', constant_values=((0, 0), (0, 0), (0, 0), (0, 0)))

        complete_output = np.zeros(np.insert(data_pad.shape[1:],0,4), dtype=data.dtype)

        z1, z2, x1, x2, y1, y2 = dataset['bbox_tumor']

        start_points = itertools.product(range(z1,z2,stride), range(x1,x2,stride), range(y1,y2,stride))

        for z,x,y in start_points:
            z_start = z - pad_net_shape[0] + pad
            z_end = z + pad_net_shape[0] + output_shape[0] + pad
            x_start = x - pad_net_shape[1] + pad
            x_end = x + pad_net_shape[1] + output_shape[1] + pad
            y_start = y - pad_net_shape[2] + pad
            y_end = y + pad_net_shape[2] + output_shape[2] + pad

            input = data_pad[:5,z_start:z_end,x_start:x_end,y_start:y_end]
            input = np.expand_dims(input, axis=0)
            input_t = torch.from_numpy(input)
            input_t = Variable(input_t).cuda()

            with torch.no_grad():
                output_t = net(input_t)

            output = output_t.squeeze_(0).data.cpu().numpy()

            complete_output[:,z+pad:z+output_shape[0]+pad,x+pad:x+output_shape[1]+pad,y+pad:y+output_shape[2]+pad] = \
                complete_output[:,z + pad:z + output_shape[0] + pad, x + pad:x + output_shape[1] + pad, y + pad:y + output_shape[2] + pad] + output

        prediction = np.argmax(complete_output,axis=0)
        prediction = prediction[pad:-pad,pad:-pad,pad:-pad]
        prediction[prediction==3]=4

        # Keep only the values where the first step found a tumor
        prediction = np.multiply(prediction, data[5,:,:,:])

        prediction = to_original_shape(prediction, dataset)
        save_nifti(opj(save_location, dataset['name'] + ".nii.gz"), prediction,  dataset['affine'])

    return


def load_predict_dataset(savedir_preproc=savedir_preproc_val1, savedir_results_step1=savedir_results_val1):
    preprocs = load_brats.load_normalized_data(dir=savedir_preproc)

    # Load results from step1
    list_results = os.listdir(savedir_results_step1)
    list_results.sort()

    for idx, dataset in enumerate(preprocs):
        name = dataset['name']

        f_result_step1 = list_results[idx]
        if name != f_result_step1[:-12]:
            raise FileNotFoundError("Wrong File")

        prediction, _ = load_nifti(opj(savedir_results_step1, f_result_step1))

        #%% Keep only the biggest found region in the brain
        cc = dataset['bbox_brain']
        pred_in_shape = prediction[cc[0]:cc[1],cc[2]:cc[3],cc[4]:cc[5]]
        biggest_size = -1
        biggest_region = 0
        for region in regionprops(label(pred_in_shape)):
            if region.area > biggest_size:
                biggest_size = region.area
                biggest_region = region

        if biggest_region == 0:
            print("ATTENTION, No Tumor found in image: " + name)
            continue

        z1, x1, y1, z2, x2, y2 = biggest_region.bbox
        bbox_tumor = [z1,z2,x1,x2,y1,y2]
        dataset['bbox_tumor'] = bbox_tumor

        dataset['data'][5,:,:,:] = np.zeros(dataset['data'].shape[1:]) # if train set, delete original segmentation
        for p in biggest_region.coords:
            dataset['data'][5,p[0],p[1],p[2]] = 1

    return preprocs


def main(mode, net_location):
    model = Unet3d_nopad(in_dim=5, out_dim=4, num_filter=32)

    net = model.cuda()
    net.load_state_dict(torch.load(net_location))

    if mode == 'validation':
        data_set = load_predict_dataset(savedir_preproc_val1, savedir_results_val1)
        predict(data_set, net, savedir_results_val2)
    elif mode == 'train':
        data_set = load_predict_dataset(savedir_preproc_train1, savedir_results_train1)
        predict(data_set, net, savedir_results_train2)
    elif mode == 'test':
        data_set = load_predict_dataset(savedir_preproc_test1, savedir_results_test1)
        predict(data_set, net, savedir_results_test2)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-d", "--device", dest="cuda_device", default='0', help="set CUDA_VISIBLE_DEVICES to this flag")
    parser.add_argument("-m", "--mode", dest="mode", default='train', help="validation, test, or train")
    parser.add_argument("-n", "--net", dest="net", help="location of trained Unet")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device

    main(args.mode, args.net)
