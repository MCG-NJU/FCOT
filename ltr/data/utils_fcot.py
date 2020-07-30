from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2
import torch
import random

def gaussian_radius(det_size, min_overlap=0.7):
    height, width = det_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    return min(r1, r2, r3)

# gaussian ridus for tensor
def gaussian_radius_tensor(height, width, min_overlap=0.7):
    assert height.shape == width.shape and height.dim() == 1   # shape: [num_images]

    zeros = torch.zeros_like(height)

    a1 = 1.0
    b1 = height + width
    c1 = width.mul(height) * (1.0 - min_overlap) / (1.0 + min_overlap)
    d1 = b1 ** 2 - 4. * a1 * c1
    sq1 = (torch.where(d1>0, d1, zeros)).sqrt()
    r1 = (b1 + sq1) / 2.           # shape: [num_images]

    a2 = 4.
    b2 = 2. * (height + width)
    c2 = (1. - min_overlap) * width.mul(height)
    d2 = b2 ** 2 - 4 * a2 * c2
    sq2 = (torch.where(d2>0, d2, zeros)).sqrt()
    r2 = (b2 + sq2) / 2.

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width.mul(height)
    d3 = b3 ** 2 - 4. * a3 * c3
    sq3 = (torch.where(d3>0, d3, zeros)).sqrt()
    r3 = (b3 + sq3) / 2.

    min_r, _ = torch.min(torch.stack((r1, r2, r3), dim=0), dim=0)
    return  min_r


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_umich_gaussian(width, height, center, radius, device, k=1):
    assert center.size(0) == radius.size(0) \
           and isinstance(width, (float,int)) and isinstance(height, (float, int))
    num_images = center.size(0)
    center, radius = center.int().numpy(), radius.int().numpy()
    heatmap = np.zeros((num_images, width, height), dtype=np.float32)

    for i in range(num_images):
        diameter = 2 * radius[i] + 1
        gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

        x, y = int(center[i][0]), int(center[i][1])

        left, right = int(min(x, radius[i])), int(min(width - x, radius[i] + 1))
        top, bottom = int(min(y, radius[i])), int(min(height - y, radius[i] + 1))
        #print(x, y, left, right, top, bottom)

        masked_heatmap = heatmap[i, y - top:y + bottom, x - left:x + right]
        masked_gaussian = gaussian[radius[i] - top:radius[i] + bottom, radius[i] - left:radius[i] + right]
        if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
            np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)

    hm = torch.from_numpy(heatmap).float().to(device)
    #print("hm device: {}".format(hm.device))
    return hm


def draw_msra_gaussian(heatmap, center, sigma):
    tmp_size = sigma * 3
    mu_x = int(center[0] + 0.5)
    mu_y = int(center[1] + 0.5)
    w, h = heatmap.shape[0], heatmap.shape[1]
    ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
    br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
    if ul[0] >= h or ul[1] >= w or br[0] < 0 or br[1] < 0:
        return heatmap
    size = 2 * tmp_size + 1
    x = np.arange(0, size, 1, np.float32)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    g_x = max(0, -ul[0]), min(br[0], h) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], w) - ul[1]
    img_x = max(0, ul[0]), min(br[0], h)
    img_y = max(0, ul[1]), min(br[1], w)
    heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]] = np.maximum(
        heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]],
        g[g_y[0]:g_y[1], g_x[0]:g_x[1]])
    return heatmap


def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
