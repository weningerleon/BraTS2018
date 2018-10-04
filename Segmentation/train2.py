import torch
from torch import optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import numpy as np
from argparse import ArgumentParser
from sklearn.utils import shuffle

from loss import DiceLoss
import preproc_train2
from misc import evaluate, AverageMeter
from directories import *
from unet_nopad import Unet3d_nopad


def main(train_args, model):
    print(train_args)

    net = model.cuda()
    net.train()

    criterion = DiceLoss()

    optimizer_adam = optim.Adam(model.parameters(),
                                lr=train_args['lr'],
                                weight_decay=train_args['weight_decay'],
                                )

    if len(train_args['snapshot']) == 0:
        curr_epoch = 0
        train_args['best_record'] = {'epoch': 0, 'val_loss': 1e10, 'acc': 0, 'acc_cls': 0, 'mean_iu': 0, 'fwavacc': 0}
    else:
        print('training resumes from ' + train_args['snapshot'])
        net.load_state_dict(torch.load(opj(savedir_nets2, train_args['snapshot'] + '.pth')))
        optimizer_adam.load_state_dict(torch.load(opj(savedir_nets2, train_args['snapshot'] + '_opt.pth')))
        split_snapshot = train_args['snapshot'].split('_')
        curr_epoch = int(split_snapshot[1]) + 1
        train_args['best_record'] = {'epoch': int(split_snapshot[1]), 'val_loss': float(split_snapshot[3]),
                                     'acc': float(split_snapshot[5]), 'acc_cls': float(split_snapshot[7]),
                                     'mean_iu': float(split_snapshot[9]), 'fwavacc': float(split_snapshot[11])}

    scheduler_adam = StepLR(optimizer=optimizer_adam, step_size=1, gamma=train_args['lr_adapt'])

    inputs, labels = preproc_train2.get_trainset()

    for epoch in range(curr_epoch, train_args['epochs']):
        train(inputs[:], labels[:], net, criterion=criterion, optimizer=optimizer_adam, epoch=epoch, train_args=train_args)
        validate(inputs[:], labels[:], net, criterion, optimizer_adam, epoch, train_args)
        scheduler_adam.step()

    return 0


def train(inputs, labels, net, criterion, optimizer, epoch, train_args):
    train_loss = AverageMeter()

    inputs, labels = shuffle(inputs, labels)

    for idx in range(0,inputs.__len__()-1,2):
        input1 = inputs[idx]
        input2 = inputs[idx+1]
        label1 = labels[idx]
        label2 = labels[idx+1]

        input = np.concatenate((input1, input2), axis=0)
        label = np.concatenate((label1, label2), axis=0)

        input_t = torch.from_numpy(input)
        label_t = torch.from_numpy(label)

        N = input_t.size(0)  # batch-size
        input_t = Variable(input_t).cuda()
        label_t = Variable(label_t).cuda()

        output_t = net(input_t)

        loss = criterion(output_t, label_t)

        loss.backward()
        optimizer.step()

        train_loss.update(loss.data, N)

        if (idx) % train_args['print_freq'] == 0 or (idx + 1) % train_args['print_freq'] == 0:
            print('[epoch %d], [iter %d / %d], [train loss %.5f]' % (
                epoch, idx + 1, len(inputs), train_loss.avg
            ))


def validate(inputs, labels, net, criterion, optimizer, epoch, train_args):
    net = net.eval()

    val_loss = AverageMeter()
    inputs_all, gts_all, predictions_all = [], [], []

    for idx in range(0,inputs.__len__()-1,2):

        input1 = inputs[idx]
        input2 = inputs[idx+1]
        label1 = labels[idx]
        label2 = labels[idx+1]

        input = np.concatenate((input1, input2), axis=0)
        label = np.concatenate((label1, label2), axis=0)

        input_t = torch.from_numpy(input)
        label_t = torch.from_numpy(label)

        N = input_t.size(0)  # batch-size
        input_t = Variable(input_t).cuda()
        label_t = Variable(label_t).cuda()

        with torch.no_grad():
            output = net(input_t)

        loss = criterion(output, label_t)

        val_loss.update(loss.data, N)

        predictions = output.data.max(1)[1].squeeze_(1).squeeze_(0).cpu().numpy()
        label = np.argmax(label, axis=1)

        gts_all.append(label.squeeze())
        predictions_all.append(predictions)

    acc, acc_cls, mean_iu, fwavacc = evaluate(predictions_all, gts_all, 4)

    if mean_iu > train_args['best_record']['mean_iu']:
        train_args['best_record']['val_loss'] = val_loss.avg
        train_args['best_record']['epoch'] = epoch
        train_args['best_record']['acc'] = acc
        train_args['best_record']['acc_cls'] = acc_cls
        train_args['best_record']['mean_iu'] = mean_iu
        train_args['best_record']['fwavacc'] = fwavacc
        snapshot_name = 'epoch_%d_loss_%.5f_mean-iu_%.5f_lr_%.10f' % (
            epoch, val_loss.avg, mean_iu, optimizer.param_groups[0]['lr'])
        torch.save(net.state_dict(), os.path.join(savedir_nets2, snapshot_name + '.pth'))
        torch.save(optimizer.state_dict(), os.path.join(savedir_nets2, snapshot_name + '_opt.pth'))
        torch.save(net.state_dict(), os.path.join(savedir_nets2, 'bestnet.pth'))
        torch.save(optimizer.state_dict(), os.path.join(savedir_nets2, 'bestnet_opt.pth'))

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
    'lr': 0.005,
    'weight_decay': 1e-5,
    'snapshot': '',  # empty string denotes learning from scratch
    'lr_adapt': 0.985,
    'lr_adapt_freq': 1,
    'print_freq': 1000,
    'val_save_to_img_file': False,
}


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-d", "--device", dest="cuda_device", default='0', help="set CUDA_VISIBLE_DEVICES to this flag")
    parser.add_argument("-s", "--snapshot", dest="snapshot", default='', help="snapshot name to continue training")

    cmdline_args = parser.parse_args()
    args['snapshot'] = cmdline_args.snapshot

    os.environ["CUDA_VISIBLE_DEVICES"] = cmdline_args.cuda_device
    model = Unet3d_nopad(in_dim=5, out_dim=4, num_filter=32)

    main(train_args=args, model=model)