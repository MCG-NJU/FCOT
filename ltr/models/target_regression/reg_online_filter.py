import torch
import torch.nn as nn
from ltr.external.PreciseRoIPooling.pytorch.prroi_pool import PrRoIPool2D
import ltr.models.layers.filter as filter_layer
import math
from ltr.external.DCNv2.dcn_v2 import DCN
import torch.nn.functional as F

class RegFilter(nn.Module):
    def __init__(self,
                 pool_size=5,
                 filter_dim=4,
                 filter_channel=256,
                 inner_channel=256,
                 input_features_size=72,
                 input_features_channel=256,
                 filter_optimizer=None,
                 train_reg_optimizer=False,
                 train_cls_72_and_reg_init=True):
        super(RegFilter, self).__init__()
        self.pool_size = pool_size
        self.filter_channel = filter_channel
        self.filter_dim=filter_dim
        self.input_features_size = input_features_size
        self.input_features_channel = input_features_channel
        self.filter_optimizer = filter_optimizer
        self.train_cls_72_and_reg_init = train_cls_72_and_reg_init

        self.reg_initializer_72 = nn.Sequential(
            nn.Conv2d(input_features_channel, inner_channel, 3, 1, 1, bias=False),
            nn.GroupNorm(32, inner_channel),
            nn.ReLU(),

            DCN(inner_channel, inner_channel, kernel_size=(3, 3), stride=1, padding=1, dilation=1, deformable_groups=1),
            nn.GroupNorm(32, inner_channel),
            nn.ReLU(),

            DCN(inner_channel, filter_channel * filter_dim, kernel_size=(3, 3), stride=1, padding=1, dilation=1,
                deformable_groups=1),
            nn.ReLU(),
        )
        self.prroipool_72 = PrRoIPool2D(pool_size, pool_size, 72 / 288.0)

        self.reg_initializer_36 = nn.Sequential(
            nn.Conv2d(input_features_channel, inner_channel, 3, 1, 1, bias=False),
            nn.GroupNorm(32, inner_channel),
            nn.ReLU(),
            DCN(inner_channel, inner_channel, kernel_size=(3, 3), stride=1, padding=1, dilation=1, deformable_groups=1),
            nn.GroupNorm(32, inner_channel),
            nn.ReLU(),

            DCN(inner_channel, filter_channel * filter_dim, kernel_size=(3, 3), stride=1, padding=1, dilation=1, deformable_groups=1),
            nn.ReLU(),
        )
        self.prroipool_36 = PrRoIPool2D(pool_size, pool_size, 36 / 288.0)
        self.reg_initializer_merge = nn.Conv2d(filter_channel * filter_dim * 2, filter_channel * filter_dim, 1, 1, bias=False)

        self.reg_head_72 = nn.Sequential(
            nn.Conv2d(input_features_channel, inner_channel, 3, 1, 1, bias=False),
            nn.GroupNorm(32, inner_channel),
            nn.ReLU(),
            DCN(inner_channel, inner_channel, kernel_size=(3, 3), stride=1, padding=1, dilation=1, deformable_groups=1),
            nn.GroupNorm(32, inner_channel),
            nn.ReLU(),

            DCN(inner_channel, filter_channel, kernel_size=(3, 3), stride=1, padding=1, dilation=1,
                deformable_groups=1),
            nn.ReLU(),
        )
        self.reg_head_36 = nn.Sequential(
            nn.Conv2d(input_features_channel, inner_channel, 3, 1, 1, bias=False),
            nn.GroupNorm(32, inner_channel),
            nn.ReLU(),
            DCN(inner_channel, inner_channel, kernel_size=(3, 3), stride=1, padding=1, dilation=1, deformable_groups=1),
            nn.GroupNorm(32, inner_channel),
            nn.ReLU(),

            DCN(inner_channel, filter_channel, kernel_size=(3, 3), stride=1, padding=1, dilation=1,
                deformable_groups=1),
            nn.ReLU(),
        )
        self.reg_head_merge = nn.Conv2d(filter_channel * 2, filter_channel, 1, 1)

        if train_reg_optimizer:
            for func_name in ['reg_initializer_36', 'reg_initializer_72', 'reg_initializer_merge',
                              'reg_head_36', 'reg_head_72', 'reg_head_merge']:
                for p in getattr(self, func_name).parameters():
                    p.requires_grad_(False)

        # Init weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        w = self.reg_initializer_merge.weight.data
        for i in range(w.size(0)):
            for j in range(w.size(1)):
                if i == j or w.size(0) + i == j:
                    w[i, j, 0, 0] = 0.5
                else:
                    w[i, j, 0, 0] = 0.0

        w = self.reg_head_merge.weight.data
        for i in range(w.size(0)):
            for j in range(w.size(1)):
                if i == j or w.size(0) + i == j:
                    w[i, j, 0, 0] = 0.5
                else:
                    w[i, j, 0, 0] = 0.0

    def forward(self, feat_train_36, feat_test_36, feat_train_72, feat_test_72, bb):
        num_images = bb.shape[0] if bb.dim() == 3 else 1
        num_sequences = bb.shape[1] if bb.dim() == 3 else 1

        # Extract initial filter using first train sample
        bb_pool = bb[0, ...].clone().view(-1, 4)
        feat_target_36 = feat_train_36.view(num_images, -1, *feat_train_36.shape[-3:])[0, ...]  # [num_sequences, 256, 36, 36]
        feat_target_72 = feat_train_72.view(num_images, -1, *feat_train_72.shape[-3:])[0, ...]  # [num_sequences, 256, 72, 72]
        init_filter = self.generate_init_filter(feat_target_36, feat_target_72, bb_pool)

        if self.train_cls_72_and_reg_init:
            feat_test_reg = self.extract_regression_feat(feat_36=feat_test_36, feat_72=feat_test_72, num_sequences=num_sequences)
            offset_maps = self.regress(init_filter, feat_test_reg)
        else:
            feat_train_reg = self.extract_regression_feat(feat_36=feat_train_36, feat_72=feat_train_72, num_sequences=num_sequences)
            feat_test_reg = self.extract_regression_feat(feat_36=feat_test_36, feat_72=feat_test_72, num_sequences=num_sequences)
            filter, filter_iter, losses = self.generate_filter_optimizer(init_filter, feat_train_reg, bb.view(-1, 4).clone())
            offset_maps = [self.regress(f, feat_test_reg) for f in filter_iter]

        return offset_maps


    def generate_filter_optimizer(self, init_filter, feat, bb, sample_weight=None, *args, **kwargs):
        weights = init_filter
        if self.filter_optimizer is not None:
            weights, weights_iter, losses = self.filter_optimizer(init_filter, feat=feat, bb=bb, sample_weight=sample_weight, *args, **kwargs)
        else:
            weights_iter = [init_filter]
            losses = None

        return weights, weights_iter, losses


    def regress(self, weights, feat):
        """Run regressor (filter) on the features (feat)."""

        offset_maps = filter_layer.apply_filter(feat, weights)
        offset_maps = torch.relu(offset_maps)

        return offset_maps


    def extract_regression_feat(self, feat_36, feat_72, num_sequences=None):
        x_36 = self.reg_head_36(feat_36)
        x_36 =  F.interpolate(x_36, size=(72, 72))
        x_72 = self.reg_head_72(feat_72)
        x_72 = F.interpolate(x_72, size=(72, 72))
        x = torch.cat((x_36, x_72), dim=1)
        x = self.reg_head_merge(x)

        if num_sequences is None:
            return x

        x = x.view(-1, num_sequences, x.shape[-3], x.shape[-2], x.shape[-1])
        return x


    def generate_init_filter(self, feat_train_36, feat_train_72, bb):
        bb_pool = bb.view(-1, 4).clone()
        # Add batch_index to rois
        batch_size = bb_pool.shape[0]
        batch_index = torch.arange(batch_size, dtype=torch.float32).view(-1, 1).to(bb_pool.device)
        # input bb is in format xywh, convert it to x0y0x1y1 format
        bb_pool[:, 2:4] = bb_pool[:, 0:2] + bb_pool[:, 2:4]
        target_roi = torch.cat((batch_index, bb_pool), dim=1)

        feat_train_36 = feat_train_36.view(-1, *feat_train_36.shape[-3:])
        feat_pool = self.reg_initializer_36(feat_train_36)
        filter_36 = self.prroipool_36(feat_pool, target_roi)

        feat_train_72 = feat_train_72.view(-1, *feat_train_72.shape[-3:])
        feat_pool_72 = self.reg_initializer_72(feat_train_72)
        filter_72 = self.prroipool_72(feat_pool_72, target_roi)

        filter_merge = torch.cat((filter_36, filter_72), dim=1)
        filter = self.reg_initializer_merge(filter_merge)

        filter = filter.view(-1, self.filter_dim, filter.shape[-3] // self.filter_dim, filter.shape[-2], filter.shape[-1])
        return filter