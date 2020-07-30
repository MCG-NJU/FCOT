import torch.nn as nn
import torch
import ltr.models.layers.filter as filter_layer
import math


class RegSteepestDescentGN(nn.Module):
    """Optimizer module for regression branch.
    It unrolls the steepest descent with Gauss-Newton iterations to optimize the target filter.
    Moreover it learns parameters in the loss itself, as described in the DiMP paper.
    args:
        num_iter:  Number of default optimization iterations.
        feat_stride:  The stride of the input feature.
        init_step_length:  Initial scaling of the step length (which is then learned).
        init_filter_reg:  Initial filter regularization weight (which is then learned).
        min_filter_reg:  Enforce a minimum value on the regularization (helps stability sometimes).
        detach_length:  Detach the filter every n-th iteration. Default is to never detech, i.e. 'Inf'."""

    def __init__(self, num_iter=1, feat_stride=4, init_step_length=1.0,
                 init_filter_reg=1e-2, min_filter_reg=1e-3, detach_length=float('Inf')):
        super().__init__()

        self.num_iter = num_iter
        self.feat_stride = feat_stride
        self.log_step_length = nn.Parameter(math.log(init_step_length) * torch.ones(1))
        self.filter_reg = nn.Parameter(init_filter_reg * torch.ones(1))
        self.min_filter_reg = min_filter_reg
        self.detach_length = detach_length


    def forward(self, weights, feat, bb, radius=0, dim=4, sample_weight=None, num_iter=None, compute_losses=True):
        """Runs the optimizer module.
        args:
            weights:  Initial weights. Dims (sequences, feat_dim, wH, wW).
            feat:  Input feature maps. Dims (images_in_sequence, [sequences], feat_dim, H, W).
            bb:  Target bounding boxes (x, y, w, h) in the image coords. Dims (images_in_sequence, [sequences], 4).
            radius: The size of vicinity of the target center.
            dim: Dims of offset maps, default is 4, indicating the distance from the center to four sides of the target.
            num_iter:  Number of iterations to run.
            compute_losses:  Whether to compute the (train) loss in each iteration.
        returns:
            weights:  The final oprimized weights.
            weight_iterates:  The weights computed in each iteration (including initial input and final output).
            losses:  Train losses."""

        # Sizes
        num_iter = self.num_iter if num_iter is None else num_iter
        num_images = feat.shape[0]
        num_sequences = feat.shape[1] if feat.dim() == 5 else 1
        filter_sz = (weights.shape[-2], weights.shape[-1])
        output_sz = (feat.shape[-2] + (weights.shape[-2] + 1) % 2, feat.shape[-1] + (weights.shape[-1] + 1) % 2)

        # Get learnable scalars
        step_length_factor = torch.exp(self.log_step_length)
        reg_weight = (self.filter_reg*self.filter_reg).clamp(min=self.min_filter_reg**2)
        # print("filter_reg: {}".format(self.filter_reg))
        # print("log_step_length: {}".format(self.log_step_length))

        w2h2_label, label_mask = self.generate_w2h2_label(bb, num_images, num_sequences, radius=radius, output_sz=output_sz, dim=dim)
        # shape: (num_images, num_sequences, 4, 72, 72)

        # Get total sample weights
        if sample_weight is None:
            sample_weight = math.sqrt(1.0 / num_images)
        elif isinstance(sample_weight, torch.Tensor):
            sample_weight = sample_weight.sqrt().view(num_images, num_sequences, 1,  1, 1)

        weight_iterates = [weights]
        losses = []

        for i in range(num_iter):
            if i > 0 and i % self.detach_length == 0:
                weights = weights.detach()

            # Compute residuals
            # feat shape: [num_images, num_sequences, 256, 72, 72], weights shape: [num_sequences, 4, 256, 5, 5]
            scores = filter_layer.apply_filter(feat, weights)
            residuals = sample_weight * label_mask * (scores - w2h2_label.detach())

            if compute_losses:
                losses.append((residuals**2).mean())

            # Compute gradient
            residuals_mapped = sample_weight * residuals
            weights_grad = filter_layer.apply_feat_transpose(feat, residuals_mapped, filter_sz, training=self.training) + \
                          reg_weight * weights
            # print("weights_grad shape: {}".format(weights_grad.shape))   # [num_sequences, 4, 256, 5, 5]

            # Map the gradient with the Jacobian
            scores_grad = filter_layer.apply_filter(feat, weights_grad)
            scores_grad = sample_weight * scores_grad
            # print("scores_grad shape: {}".format(scores_grad.shape))    # [num_images, num_sequences, 4, 72, 72]

            # Compute optimal step length
            alpha_num = (weights_grad * weights_grad).view(num_sequences, -1).sum(dim=1)
            alpha_den = ((scores_grad * scores_grad).view(num_images, num_sequences, -1).sum(dim=(0,2)) + reg_weight * alpha_num).clamp(1e-8)
            # print("alpha_num: {}, alpha_den: {}".format(alpha_num, alpha_den))
            alpha = alpha_num / alpha_den

            # Update filter
            weights = weights - (step_length_factor * alpha.view(-1, 1, 1, 1, 1)) * weights_grad

            # Add the weight iterate
            weight_iterates.append(weights)

        if compute_losses:
            scores = filter_layer.apply_filter(feat, weights)
            losses.append(((sample_weight * label_mask * (scores - w2h2_label.detach()))**2).mean())

        return weights, weight_iterates, losses


    def generate_w2h2_label(self, bb, num_images, num_sequences, radius=0, output_sz=(72, 72), dim=4):
        """
            args:
                bb:  Target bounding boxes (x, y, w, h) in the image coords. Dims (images_in_sequence, [sequences], 4).
        """
        bb_t = bb.clone().view(-1, 4)
        center = ((bb_t[..., :2] + bb_t[..., 2:] / 2) / self.feat_stride).int().float()
        l, t = (bb_t[..., 0] / self.feat_stride).int().float(), (bb_t[..., 1] / self.feat_stride).int().float()
        r, b = ((bb_t[..., 0] + bb_t[..., 2]) / self.feat_stride).int().float(), ((bb_t[..., 1] + bb_t[..., 3]) / self.feat_stride).int().float()
        w2h2_center = torch.stack((center[..., 0] - l, r - center[..., 0], center[..., 1] - t, b - center[..., 1]), dim=1)
        # print("w2h2_center shape: {}".format(w2h2_center.shape))  # [num_images * num_sequences, 4]

        width = output_sz[0]
        w2h2_label = torch.zeros(num_images * num_sequences, width, width, dim).to(bb.device)
        label_mask = torch.zeros_like(w2h2_label).to(bb.device)

        for r_w in range(-1 * radius, radius + 1):
            for r_h in range(-1 * radius, radius + 1):
                wl = w2h2_center[..., 0] + r_w
                wr = w2h2_center[..., 1] - r_w
                ht = w2h2_center[..., 2] + r_h
                hb = w2h2_center[..., 3] - r_h

                pos = torch.stack((center[..., 0] + r_w, center[..., 1] + r_h), dim=1)
                if (wl <= 0.).any() or (wr <= 0.).any() or (ht <= 0.).any() or (hb <= 0.).any():
                    continue
                if (pos < 0.).any() or (pos >= 1.0 * width).any():
                    continue

                w2h2_cur = torch.stack((wl, wr, ht, hb), dim=1)  # (num_images * num_sequences, 4)
                index = (torch.arange(w2h2_cur.size(0)).long(), pos[..., 0].long(), pos[..., 1].long())
                w2h2_label.index_put_(index, w2h2_cur)
                label_mask.index_put_(index, torch.tensor([1.0]).to(bb.device))
                # print("w2h2_cur shape: {}".format(w2h2_cur.shape))   # [num_images * num_sequences, 4]

        w2h2_label = w2h2_label.permute(0, 3, 1, 2).view(num_images, num_sequences, dim, width, width)
        label_mask = label_mask.permute(0, 3, 1, 2).view(num_images, num_sequences, dim, width, width)
        return w2h2_label, label_mask