import torch
from torch import nn

class IOULoss(nn.Module):
    def forward(self, pred, target, weight=None):
        pred_left = pred[:, 0]
        pred_top = pred[:, 2]
        pred_right = pred[:, 1]
        pred_bottom = pred[:, 3]

        target_left = target[:, 0]
        target_top = target[:, 2]
        target_right = target[:, 1]
        target_bottom = target[:, 3]

        target_area = (target_left + target_right) * \
                      (target_top + target_bottom)
        pred_area = (pred_left + pred_right) * \
                    (pred_top + pred_bottom)

        w_intersect = torch.min(pred_left, target_left) + \
                      torch.min(pred_right, target_right)
        h_intersect = torch.min(pred_bottom, target_bottom) + \
                      torch.min(pred_top, target_top)

        area_intersect = w_intersect * h_intersect
        area_union = target_area + pred_area - area_intersect

        losses = -torch.log((area_intersect + 1.0) / (area_union + 1.0))

        if weight is not None and weight.sum() > 0:
            return (losses * weight).sum() / weight.sum()
        else:
            assert losses.numel() != 0
            return losses.mean()


class L2RegLoss(nn.Module):
    def __init__(self):
        super(L2RegLoss, self).__init__()
        self.loss = nn.MSELoss()

    def forward(self, pred, target):
        losses = self.loss(pred, target)
        return losses


class REGLoss(nn.Module):
    def __init__(self, dim=4, loss_type='iou'):
        super(REGLoss, self).__init__()
        self.dim = dim
        assert loss_type == 'l2' or loss_type == 'iou'
        if loss_type == 'iou':
            self.loss = IOULoss()
        else:
            self.loss = L2RegLoss()

    def forward(self, output, ind, target, radius=None, stride=None):
        width, height = output.size(-2), output.size(-1)
        output = output.view(-1, self.dim, width, height)
        # mask =  mask.view(-1, 2)
        target = target.view(-1, self.dim)
        ind = ind.view(-1, 1)
        center_w = (ind % width).int().float()
        center_h = (ind / width).int().float()

        if stride is None:
            stride = 288.0 / width

        # regress for the coordinates in the vicinity of the target center, the default radius is 2.
        if radius is not None:
            loss = []
            for r_w in range(-1 * radius, radius + 1):
                for r_h in range(-1 * radius, radius + 1):
                    target_wl = target[:, 0] + r_w
                    target_wr = target[:, 1] - r_w
                    target_ht = target[:, 2] + r_h
                    target_hb = target[:, 3] - r_h
                    if (target_wl < 0.).any() or (target_wr < 0.).any() or (target_ht < 0.).any() or (target_hb < 0.).any():
                        continue
                    if (center_h + r_h < 0.).any() or (center_h + r_h >= 1.0 * width).any() \
                            or (center_w + r_w < 0.).any() or (center_w + r_w >= 1.0 * width).any():
                        continue

                    target_curr = torch.stack((target_wl, target_wr, target_ht, target_hb), dim=1)  # [num_images * num_sequences, 4]
                    ind_curr = ((center_h + r_h) * width + (center_w + r_w)).long()
                    pred_curr = _tranpose_and_gather_feat(output, ind_curr)
                    loss_curr = self.loss(pred_curr, target_curr)
                    loss.append(loss_curr)
            if len(loss) == 0:
                pred = _tranpose_and_gather_feat(output, ind.long())  # pred shape: [num_images * num_sequences, 4]
                loss = self.loss(pred, target)
                return loss
            loss = torch.stack(loss, dim=0)
            loss = torch.mean(loss, dim=0)
            return loss
        pred = _tranpose_and_gather_feat(output, ind.long())     # pred shape: [num_images * num_sequences, 4]
        loss = self.loss(pred, target)

        return loss

def _tranpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat

def _gather_feat(feat, ind, mask=None):
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(dim=1, index=ind)   # [num_images * num_sequences, 1, 2]
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat.view(ind.size(0), dim)


class IOUPred(nn.Module):
    def forward(self, pred, target):
        pred_left = pred[:, 0]
        pred_top = pred[:, 2]
        pred_right = pred[:, 1]
        pred_bottom = pred[:, 3]

        target_left = target[:, 0]
        target_top = target[:, 2]
        target_right = target[:, 1]
        target_bottom = target[:, 3]
        # print(pred_left, pred_right, pred_top, pred_bottom)

        target_area = (target_left + target_right) * \
                      (target_top + target_bottom)
        pred_area = (pred_left + pred_right) * \
                    (pred_top + pred_bottom)

        w_intersect = torch.min(pred_left, target_left) + \
                      torch.min(pred_right, target_right)
        h_intersect = torch.min(pred_bottom, target_bottom) + \
                      torch.min(pred_top, target_top)

        area_intersect = w_intersect * h_intersect
        area_union = target_area + pred_area - area_intersect

        iou = (area_intersect + 1.0) / (area_union + 1.0)

        assert iou.numel() != 0
        return iou.mean()


class DenseIouPred(nn.Module):
    def __init__(self, dim=4):
        super(DenseIouPred, self).__init__()
        self.dim = dim
        self.iou_pred = IOUPred()

    def forward(self, output, ind, target, radius=10):
        """
        :param output: shape: (num_images, num_sequences, 4, 72, 72)
        :param ind: shape: (num_images, num_sequences, 1)
        :param target: shape: (num_images, num_sequences, 4)
        """
        width, height = output.size(-2), output.size(-1)
        output = output.view(-1, self.dim, width, height)[0, ...].unsqueeze(0)
        target = target.view(-1, self.dim)[0, ...].unsqueeze(0)
        ind = ind.view(-1, 1)[0, ...].unsqueeze(0)

        center_w = (ind % width).int().float()
        center_h = (ind / width).int().float()

        iou_map = torch.zeros(width, height).to(output.device)

        assert radius is not None
        for r_w in range(-1 * radius, radius + 1):
            for r_h in range(-1 * radius, radius + 1):
                target_wl = target[:, 0] + r_w
                target_wr = target[:, 1] - r_w
                target_ht = target[:, 2] + r_h
                target_hb = target[:, 3] - r_h
                if (target_wl < 0.).any() or (target_wr < 0.).any() or (target_ht < 0.).any() or (target_hb < 0.).any():
                    continue
                if (center_h + r_h < 0.).any() or (center_h + r_h >= 1.0 * width).any() \
                        or (center_w + r_w < 0.).any() or (center_w + r_w >= 1.0 * width).any():
                    continue

                target_curr = torch.stack((target_wl, target_wr, target_ht, target_hb), dim=1)  # [num_images * num_sequences, 4]
                ind_curr = ((center_h + r_h) * width + (center_w + r_w)).long()
                pred_curr = _tranpose_and_gather_feat(output, ind_curr)
                iou_curr = self.iou_pred(pred_curr, target_curr)
                iou_map[int(center_h.item() + r_h)][int(center_w.item() + r_w)] = iou_curr.item()

        return iou_map