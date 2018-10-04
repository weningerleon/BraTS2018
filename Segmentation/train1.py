import torch
from torch import optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import load_brats
import numpy as np
import torch.nn as nn
import random
from argparse import ArgumentParser

from misc import evaluate, AverageMeter
from directories import *
from unet3d import Unet3d


def main(train_args, model):
    print(train_args)

    net = model.cuda()
    net.train()

    train_set = load_brats.load_normalized_data(ids=0)
    random.shuffle(train_set)

    criterion = nn.BCEWithLogitsLoss().cuda()

    optimizer_adam = optim.Adam(model.parameters(),
                                lr=train_args['lr'],
                                weight_decay=train_args['weight_decay'],
                                )

    if len(train_args['snapshot']) == 0:
        curr_epoch = 0
        train_args['best_record'] = {'epoch': 0, 'val_loss': 1e10, 'acc': 0, 'acc_cls': 0, 'mean_iu': 0, 'fwavacc': 0}
    else:
        print('training resumes from ' + train_args['snapshot'])
        net.load_state_dict(torch.load(opj(savedir_nets1, train_args['snapshot'] + '.pth')))
        optimizer_adam.load_state_dict(torch.load(opj(savedir_nets1, train_args['snapshot'] + '_opt.pth')))
        split_snapshot = train_args['snapshot'].split('_')
        curr_epoch = int(split_snapshot[3]) + 1
        train_args['best_record'] = {'epoch': int(split_snapshot[3]), 'val_loss': float(split_snapshot[5]),
                                     'acc': float(split_snapshot[7]), 'acc_cls': float(split_snapshot[9]),
                                     'mean_iu': float(split_snapshot[11]), 'fwavacc': float(split_snapshot[13])}

    scheduler_adam = StepLR(optimizer=optimizer_adam, step_size=1, gamma=train_args['lr_adapt'])

    train_set = load_brats.load_normalized_data()

    for epoch in range(curr_epoch, train_args['epochs']):
        train(train_set, net, criterion=criterion, optimizer=optimizer_adam, epoch=epoch, train_args=train_args)
        validate(train_set, net, criterion, optimizer_adam, epoch, train_args)
        scheduler_adam.step()

    return 0


def train(train_sets, net, criterion, optimizer, epoch, train_args):
    train_loss = AverageMeter()
    cur_iter = 0

    random.shuffle(train_sets)

    for train_set in train_sets:

        data = train_set['data']

        datashape = data.shape[1:]

        zeropad_shape = np.ceil(np.divide(datashape, 8)).astype(np.int)*8
        p = zeropad_shape - datashape  # padding
        p_b = np.ceil(p/2).astype(np.int)  # padding before image
        p_a = np.floor(p/2).astype(np.int)  # padding after image

        data = np.pad(data, ((0,0),(p_b[0],p_a[0]),(p_b[1],p_a[1]),(p_b[2],p_a[2])),
                      mode='constant', constant_values=((0,0),(0,0),(0,0),(0,0)))

        inputs = data[:5,:,:,:]
        inputs = np.expand_dims(inputs, axis=0)

        labels = data[5:6,:,:,:]
        labels[labels!=0]=1 # Find just the tumor
        labels = np.int64(labels)
        labels = np.eye(2)[labels]
        labels = np.moveaxis(labels, -1, 1)
        labels = np.float32(labels)
        inputs = torch.from_numpy(inputs)
        labels = torch.from_numpy(labels)

        N = inputs.size(0)  # batch-size
        inputs = Variable(inputs).cuda()
        labels = Variable(labels).cuda()

        outputs = net(inputs)

        loss = criterion(outputs, labels) / N
        loss.backward()
        optimizer.step()

        train_loss.update(loss.data, N)

        if (cur_iter) % train_args['print_freq'] == 0:
            print('[epoch %d], [iter %d / %d], [train loss %.5f]' % (
                epoch, cur_iter, len(train_sets), train_loss.avg
            ))
        cur_iter += 1


def validate(val_sets, net, criterion, optimizer, epoch, train_args):
    net.eval()

    val_loss = AverageMeter()
    inputs_all, gts_all, predictions_all = [], [], []

    for val_set in val_sets:

        data = val_set['data']
        datashape = data.shape[1:]

        zeropad_shape = np.ceil(np.divide(datashape, 8)).astype(np.int) * 8
        p = zeropad_shape - datashape  # padding
        p_b = np.ceil(p / 2).astype(np.int)  # padding before image
        p_a = np.floor(p / 2).astype(np.int)  # padding after image

        data_pad = np.pad(data, ((0, 0), (p_b[0], p_a[0]), (p_b[1], p_a[1]), (p_b[2], p_a[2])),
                      mode='constant', constant_values=((0, 0), (0, 0), (0, 0), (0, 0)))

        inputs = data_pad[:5,:,:,:] # just use t1 & flair
        inputs = np.expand_dims(inputs, axis=0)

        labels = data_pad[5:6,:,:,:]
        labels[labels!=0]=1
        labels = np.int64(labels)
        labels = np.eye(2)[labels]
        labels = np.moveaxis(labels, -1, 1)
        labels = np.float32(labels)

        inputs = torch.from_numpy(inputs)
        labels = torch.from_numpy(labels)

        N = inputs.size(0)  # batch-size
        inputs = Variable(inputs).cuda()
        labels = Variable(labels).cuda()

        with torch.no_grad():
            outputs = net(inputs)

        loss = criterion(outputs, labels) / N

        val_loss.update(loss.data, N)

        predictions = outputs.data.max(1)[1].squeeze_(1).squeeze_(0).cpu().numpy()
        p_up = predictions.shape - p_a
        predictions = predictions[p_b[0]:p_up[0], p_b[1]:p_up[1], p_b[2]:p_up[2]]

        gts_all.append(data[5:6,:,:,:].squeeze())
        predictions_all.append(predictions)

    acc, acc_cls, mean_iu, fwavacc = evaluate(predictions_all, gts_all, N)

    if mean_iu > train_args['best_record']['mean_iu']:
        train_args['best_record']['val_loss'] = val_loss.avg
        train_args['best_record']['epoch'] = epoch
        train_args['best_record']['acc'] = acc
        train_args['best_record']['acc_cls'] = acc_cls
        train_args['best_record']['mean_iu'] = mean_iu
        train_args['best_record']['fwavacc'] = fwavacc
        snapshot_name = 'epoch_%d_loss_%.5f_mean-iu_%.5f_lr_%.10f' % (
            epoch, val_loss.avg, mean_iu, optimizer.param_groups[0]['lr'])
        torch.save(net.state_dict(), os.path.join(savedir_nets1, 'bestnet.pth'))
        torch.save(optimizer.state_dict(), os.path.join(savedir_nets1, 'bestnet_opt.pth'))

        torch.save(net.state_dict(), os.path.join(savedir_nets1, snapshot_name + '.pth'))
        torch.save(optimizer.state_dict(), os.path.join(savedir_nets1, snapshot_name + '_opt.pth'))

    print('--------------------------------------------------------------------')
    print('[epoch %d], [val loss %.5f], [acc %.5f], [acc_cls %.5f], [mean_iu %.5f], [fwavacc %.5f]' % (
        epoch, val_loss.avg, acc, acc_cls, mean_iu, fwavacc))
    print('best record: [val loss %.5f], [acc %.5f], [acc_cls %.5f], [mean_iu %.5f], [fwavacc %.5f], [epoch %d]' % (
        train_args['best_record']['val_loss'], train_args['best_record']['acc'], train_args['best_record']['acc_cls'],
        train_args['best_record']['mean_iu'], train_args['best_record']['fwavacc'], train_args['best_record']['epoch']))
    print('--------------------------------------------------------------------')

    net.train()
    return



########################################################################################################################

args = {
    'epochs': 300,
    'lr': 0.001,
    'weight_decay': 1e-5,
    'snapshot': '',  # empty string denotes learning from scratch
    'lr_adapt': 0.985,
    'lr_adapt_freq': 1,
    'print_freq': 100,
    'val_save_to_img_file': False,
}


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-d", "--device", dest="cuda_device", default='0', help="set CUDA_VISIBLE_DEVICES to this flag")
    parser.add_argument("-s", "--snapshot", dest="snapshot", default='', help="snapshot name to continue training")

    cmdline_args = parser.parse_args()
    args['snapshot'] = cmdline_args.snapshot

    os.environ["CUDA_VISIBLE_DEVICES"] = cmdline_args.cuda_device
    model = Unet3d(in_dim=5, out_dim=2, num_filter=16)

    main(train_args=args, model=model)
