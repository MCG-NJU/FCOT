from torch import nn
from torchvision.models.resnet import BasicBlock, Bottleneck
from ltr.models.layers.normalization import InstanceL2Norm
from ltr.models.layers.transform import InterpCat
from ltr.external.DCNv2.dcn_v2 import DCN

def residual_basic_block(feature_dim=256, num_blocks=1, l2norm=True, final_conv=False, norm_scale=1.0, out_dim=None,
                         interp_cat=False):
    """Construct a network block based on the BasicBlock used in ResNet 18 and 34."""
    if out_dim is None:
        out_dim = feature_dim
    feat_layers = []
    if interp_cat:
        feat_layers.append(InterpCat())
    for i in range(num_blocks):
        odim = feature_dim if i < num_blocks - 1 + int(final_conv) else out_dim
        feat_layers.append(BasicBlock(feature_dim, odim))
    if final_conv:
        feat_layers.append(nn.Conv2d(feature_dim, out_dim, kernel_size=3, padding=1, bias=False))
    if l2norm:
        feat_layers.append(InstanceL2Norm(scale=norm_scale))
    return nn.Sequential(*feat_layers)


def residual_basic_block_pool(feature_dim=256, num_blocks=1, l2norm=True, final_conv=False, norm_scale=1.0, out_dim=None,
                              pool=True):
    """Construct a network block based on the BasicBlock used in ResNet."""
    if out_dim is None:
        out_dim = feature_dim
    feat_layers = []
    for i in range(num_blocks):
        odim = feature_dim if i < num_blocks - 1 + int(final_conv) else out_dim
        feat_layers.append(BasicBlock(feature_dim, odim))
    if final_conv:
        feat_layers.append(nn.Conv2d(feature_dim, out_dim, kernel_size=3, padding=1, bias=False))
    if pool:
        feat_layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    if l2norm:
        feat_layers.append(InstanceL2Norm(scale=norm_scale))

    return nn.Sequential(*feat_layers)


def residual_bottleneck(feature_dim=256, num_blocks=1, l2norm=True, final_conv=False, norm_scale=1.0, out_dim=None,
                        interp_cat=False):
    """Construct a network block based on the Bottleneck block used in ResNet."""
    if out_dim is None:
        out_dim = feature_dim
    feat_layers = []
    if interp_cat:
        feat_layers.append(InterpCat())
    for i in range(num_blocks):
        planes = feature_dim if i < num_blocks - 1 + int(final_conv) else out_dim // 4
        feat_layers.append(Bottleneck(4*feature_dim, planes))
    if final_conv:
        feat_layers.append(nn.Conv2d(4*feature_dim, out_dim, kernel_size=3, padding=1, bias=False))
    if l2norm:
        feat_layers.append(InstanceL2Norm(scale=norm_scale))
    return nn.Sequential(*feat_layers)


def clf_head_18(feature_dim=256, num_blocks=1, l2norm=True, final_conv=False, norm_scale=1.0, out_dim=None,
                interp_cat=False):
    if out_dim is None:
        out_dim = feature_dim
    feat_layers = []
    if interp_cat:
        feat_layers.append(InterpCat())
    for i in range(num_blocks):
        planes = feature_dim if i < num_blocks - 1 + int(final_conv) else out_dim // 4
        feat_layers.append(Bottleneck(4*feature_dim, planes))
    if final_conv:
        feat_layers.append(nn.Conv2d(4*feature_dim, out_dim, kernel_size=3, padding=1, bias=False))
    if l2norm:
        feat_layers.append(InstanceL2Norm(scale=norm_scale))
    return nn.Sequential(*feat_layers)


def clf_head_72(feature_dim=256, l2norm=True, norm_scale=1.0, inner_dim=256, out_dim=None):
    if out_dim is None:
        out_dim = feature_dim
    feat_layers = []

    feat_layers.append(nn.Conv2d(feature_dim, inner_dim, kernel_size=3, padding=1))
    feat_layers.append(nn.GroupNorm(32, inner_dim))
    feat_layers.append(nn.ReLU(inplace=True))

    feat_layers.append(DCN(inner_dim, inner_dim, kernel_size=3, stride=1, padding=1, dilation=1, deformable_groups=1))
    feat_layers.append(nn.GroupNorm(32, inner_dim))
    feat_layers.append(nn.ReLU(inplace=True))

    feat_layers.append(DCN(inner_dim, out_dim, kernel_size=3, stride=1, padding=1, dilation=1, deformable_groups=1))

    if l2norm:
        feat_layers.append(InstanceL2Norm(scale=norm_scale))
    return nn.Sequential(*feat_layers)