import torch
import torch.nn as nn


def conv_block(in_dim, out_dim, act_fn):
    model = nn.Sequential(
        nn.Conv3d(in_dim,out_dim, kernel_size=3, stride=1, padding=0),
        nn.InstanceNorm3d(out_dim),
        act_fn,
    )
    return model


def up_conv(in_dim, out_dim, act_fn):
    model = nn.Sequential(
        nn.ConvTranspose3d(in_dim,out_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.InstanceNorm3d(out_dim),
        act_fn,
    )
    return model


def double_conv_block(in_dim, out_dim, act_fn):
    model = nn.Sequential(
        conv_block(in_dim, out_dim, act_fn),
        conv_block(out_dim, out_dim, act_fn),
    )
    return model


def out_block(in_dim,out_dim):
    model = nn.Sequential(
        nn.Conv3d(in_dim,out_dim, kernel_size=1, stride=1, padding=0),
        nn.Softmax(dim=1),
    )
    return model


class Unet3d_nopad(nn.Module):

    def __init__(self, in_dim, out_dim, num_filter):
        super(Unet3d_nopad, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_filter = num_filter
        act_fn = nn.ReLU(inplace=True)

        self.block_1 = double_conv_block(self.in_dim, self.num_filter, act_fn)
        self.pool_1 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)

        self.block_2 = double_conv_block(self.num_filter, self.num_filter * 2, act_fn)
        self.pool_2 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)

        self.block_3 = double_conv_block(self.num_filter * 2, self.num_filter * 4, act_fn)
        self.pool_3 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)

        self.bridge = double_conv_block(self.num_filter * 4, self.num_filter * 8, act_fn)

        self.trans_1 = up_conv(self.num_filter * 8, self.num_filter * 4, act_fn)
        self.block_5 = double_conv_block(self.num_filter * 8, self.num_filter * 4, act_fn)

        self.trans_2 = up_conv(self.num_filter * 4, self.num_filter * 2, act_fn)
        self.block_6 = double_conv_block(self.num_filter * 4, self.num_filter * 2, act_fn)

        self.trans_3 = up_conv(self.num_filter * 2, self.num_filter * 1, act_fn)
        self.block_7 = double_conv_block(self.num_filter * 2, self.num_filter * 1, act_fn)

        self.out = out_block(self.num_filter, out_dim)


    def forward(self, x):
        block_1 = self.block_1(x)  # 124 -> 120
        pool_1 = self.pool_1(block_1)  # 120 -> 60

        block_2 = self.block_2(pool_1)  # 60 -> 56
        pool_2 = self.pool_2(block_2)  # 56 -> 28

        block_3 = self.block_3(pool_2)  # 28 -> 24
        pool_3 = self.pool_3(block_3)  # 24 -> 12

        bridge = self.bridge(pool_3)  # 12 -> 8

        trans_1 = self.trans_1(bridge)  # 8 -> 16

        concat_1 = torch.cat([trans_1, crop3d(block_3, 4)], dim=1)
        block_5 = self.block_5(concat_1)  # 16 -> 12

        trans_2 = self.trans_2(block_5)  # 12 -> 24
        concat_2 = torch.cat([trans_2, crop3d(block_2, 16)], dim=1)
        block_6 = self.block_6(concat_2)  # 24 -> 20

        trans_3 = self.trans_3(block_6)  # 20 -> 40
        concat_3 = torch.cat([trans_3, crop3d(block_1, 40)], dim=1)
        block_7 = self.block_7(concat_3)  # 40 -> 36

        out = self.out(block_7)

        return out


def crop3d(variable, d):
    # croppes the variable at all 3 sides by d
    return variable[:,:,d:-d,d:-d,d:-d]