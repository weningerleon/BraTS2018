import torch
import numpy as np
from torch.autograd import Variable
from dipy.io.image import save_nifti
from misc import to_original_shape

import load_brats
from directories import *
from unet3d import Unet3d

small_testing = False
from argparse import ArgumentParser


def predict(predict_sets, net, save_location):
    net.eval()

    for val_set in predict_sets:
        data = val_set['data']
        datashape = data.shape[1:]

        zeropad_shape = np.ceil(np.divide(datashape, 8)).astype(np.int) * 8

        p = zeropad_shape - datashape  # padding
        p_b = np.ceil(p / 2).astype(np.int)  # padding before image
        p_a = np.floor(p / 2).astype(np.int)  # padding after image

        data_pad = np.pad(data, ((0, 0), (p_b[0], p_a[0]), (p_b[1], p_a[1]), (p_b[2], p_a[2])),
                          mode='constant', constant_values=((0, 0), (0, 0), (0, 0), (0, 0)))

        inputs = data_pad[:5, :, :, :]  # just use t1, t2, flair
        inputs = np.expand_dims(inputs, axis=0)
        inputs = torch.from_numpy(inputs)
        inputs = Variable(inputs).cuda()

        with torch.no_grad():
            outputs = net(inputs)

        # Bring predictions into correct shape and remove the zero-padded voxels
        predictions = outputs.data.max(1)[1].squeeze_(1).squeeze_(0).cpu().numpy()
        p_up = predictions.shape - p_a
        predictions = predictions[p_b[0]:p_up[0], p_b[1]:p_up[1], p_b[2]:p_up[2]]

        # Set all Voxels that are outside of the brainmask to 0
        mask = (data[0, :, :, :] != 0) | (data[1, :, :, :] != 0) | (data[3, :, :, :] != 0) | (data[4, :, :, :] != 0)
        predictions = np.multiply(predictions, mask)

        pred_orig_shape = to_original_shape(predictions, val_set)
        save_nifti(opj(save_location, val_set['name'] + '_pred.nii.gz'), pred_orig_shape, val_set['affine'])
        print("Step1: %s  \tprediction complete" %val_set['name'])

    return


def main(mode, net_location):
    model = Unet3d(in_dim=5, out_dim=2, num_filter=16)

    net = model.cuda()
    net.load_state_dict(torch.load(net_location))

    if mode == 'train':
        data_set = load_brats.load_normalized_data(dir=savedir_preproc_train1)
        savedir_results = savedir_results_train1
    elif mode == 'validation':
        data_set = load_brats.load_normalized_data(dir=savedir_preproc_val1)
        savedir_results = savedir_results_val1
    elif mode == 'test':
        data_set = load_brats.load_normalized_data(dir=savedir_preproc_test1)
        savedir_results = savedir_results_test1

    predict(data_set, net, savedir_results)
    print('Step1: Finished!')


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-d", "--device", dest="cuda_device", default='0', help="set CUDA_VISIBLE_DEVICES to this flag")
    parser.add_argument("-m", "--mode", dest="mode", default='validation', help="validation, test, or train")
    parser.add_argument("-n", "--net", dest="net", help="location of trained Unet")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device

    main(args.mode, args.net)
