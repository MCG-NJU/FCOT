from pytracking.tracker.base import BaseTracker
import torch
import torch.nn.functional as F
import torch.nn as nn
import math
import time
from pytracking import dcf, TensorList
from pytracking.features.preprocessing import numpy_to_torch
from pytracking.utils.plotting import show_tensor, plot_graph
from pytracking.features.preprocessing import sample_patch_multiscale, sample_patch_transformed, sample_patch
from pytracking.features import augmentation
import cv2
import os
import numpy as np


class FcotTracker(BaseTracker):

    def initialize_features(self):
        if not getattr(self, 'features_initialized', False):
            self.params.net.initialize()
        self.features_initialized = True

    def initialize(self, image, info: dict) -> dict:
        # Initialize some stuff
        self.frame_num = 1
        if not hasattr(self.params, 'device'):
            self.params.device = 'cuda' if self.params.use_gpu else 'cpu'

        # Initialize network
        # self.initialize_features()

        # The DiMP network
        self.net = self.params.net

        # Time initialization
        tic = time.time()

        # Get target position and size
        state = info['init_bbox']
        self.pos = torch.Tensor([state[1] + (state[3] - 1)/2, state[0] + (state[2] - 1)/2])
        self.target_sz = torch.Tensor([state[3], state[2]])
        self.init_target_sz = self.target_sz

        # Set sizes
        sz = self.params.image_sample_size     # 288
        self.img_sample_sz = torch.Tensor([sz, sz] if isinstance(sz, int) else sz)
        self.img_support_sz = self.img_sample_sz   # [288, 288]

        # Set search area
        search_area = torch.prod(self.target_sz * self.params.search_area_scale).item()
        self.target_scale =  math.sqrt(search_area) / self.img_sample_sz.prod().sqrt()
        self.init_target_scale = self.target_scale

        # Target size in base scale
        self.base_target_sz = self.target_sz / self.target_scale

        # Convert image
        im = numpy_to_torch(image)

        # Setup scale factors
        if not hasattr(self.params, 'scale_factors'):
            self.params.scale_factors = torch.ones(1)
        elif isinstance(self.params.scale_factors, (list, tuple)):
            self.params.scale_factors = torch.Tensor(self.params.scale_factors)

        # Setup scale bounds
        self.image_sz = torch.Tensor([im.shape[2], im.shape[3]])
        self.min_scale_factor = torch.max(10 / self.base_target_sz)
        self.max_scale_factor = torch.min(self.image_sz / self.base_target_sz)

        # Extract and transform sample
        init_backbone_feat = self.generate_init_samples(im)

        # Initialize classifiers and regressor
        self.init_classifier_and_regressor(init_backbone_feat)

        self.prev_pos = self.pos
        self.prev_box = torch.Tensor([state[1], state[0], state[1]+state[3], state[0]+state[2]])
        self.train_x_72 = None
        self.train_x_72_cls = None
        self.train_x_18_cls = None
        self.train_x_72_reg = None
        self.target_box = None
        self.s = None
        self.max_score = -100

        out = {'time': time.time() - tic}
        return out


    def track(self, image) -> dict:
        self.debug_info = {}

        self.frame_num += 1
        self.debug_info['frame_num'] = self.frame_num
        print("---------track frame-{}-----------".format(self.frame_num))

        # Convert image to tensor [1, 3, w, h]
        im = numpy_to_torch(image)

        # ------- LOCALIZATION ------- #

        backbone_feat, sample_coords = self.extract_backbone_features(im,
                                                                      self.prev_pos,
                                                                      self.target_scale * self.params.scale_factors,
                                                                      self.img_sample_sz)

        # Extract classification features
        with torch.no_grad():
            test_x_18_cls = self.get_classification_features(backbone_feat)
            backbone_feat_ = self.net.get_backbone_clf_feat(backbone_feat)
            test_x_18_pyramid = self.net.pyramid_first_conv(x=None, x_backbone=backbone_feat_)
            test_x_36_pyramid = self.net.pyramid_36(test_x_18_pyramid, backbone_feat['layer2'])
            test_x_72_pyramid = self.net.pyramid_72(test_x_36_pyramid, backbone_feat['layer1'])

            test_x_72_cls = self.net.classifier_72.extract_classification_feat(test_x_72_pyramid)
            test_x_72_reg = self.net.regressor_72.extract_regression_feat(test_x_36_pyramid, test_x_72_pyramid)

        # Location of sample
        sample_pos, sample_scales = self.get_sample_location(sample_coords)

        # Compute classification scores
        scores_raw_18 = self.classify_target_18(test_x_18_cls)
        scores_72_1 = F.interpolate(scores_raw_18, size=(72, 72))
        scores_raw_72 = self.classify_target_72(test_x_72_cls)
        scores_72_2 = F.interpolate(scores_raw_72, size=(72, 72))
        scores_72 = scores_72_1 * self.params.merge_rate_18 + scores_72_2 * self.params.merge_rate_72

        # locate the center pos
        translation_vec, scale_ind, s, flag, max_disp, max_disp2, max_score = \
                                    self.localize_target(scores_72, sample_scales, feature_sz=72)
        new_pos = sample_pos[scale_ind, :] + translation_vec

        max_disp = max_disp.long()
        max_disp2 = max_disp2.long()

        # w2h2 predicticn
        with torch.no_grad():
            w2h2_72_cur = self.net.regressor_72.regress(self.target_reg_filter_72, test_x_72_reg)
            w2h2_72_init = self.net.regressor_72.regress(self.init_reg_filter, test_x_72_reg)
            w2h2_72 = w2h2_72_cur * self.params.reg_lamda + w2h2_72_init * (1.0 - self.params.reg_lamda)

        # get the distance from the center_pos to the target sides.
        w2h2_prediction_72 = self.ctdet_decode(w2h2_72, max_disp).squeeze().to(self.img_sample_sz.device)
        w2h2_in_init = w2h2_prediction_72 * sample_scales[scale_ind]

        # box: (y0, x0, y1, x1)
        box = torch.tensor([new_pos[0] - w2h2_in_init[2],
                            new_pos[1] - w2h2_in_init[0],
                            new_pos[0] + w2h2_in_init[3],
                            new_pos[1] + w2h2_in_init[1]]).clamp(min=1, max=torch.max(self.image_sz)).to(new_pos.device)

        if self.params.iou_select and self.prev_box is not None:
            iou = self.iou_pred(box, self.prev_box)
            if iou < 0.05 and flag != 'not_found':
                if flag == 'normal':
                    flag = 'uncertain'
                else:
                    flag = 'not_found'

        ###### For Visualization ######
        # w2h2_288 = w2h2_prediction_72.clone()
        # center_288 = max_disp * 4 + 2
        # box_288 = torch.tensor([center_288[0] - w2h2_288[2],
        #                         center_288[1] - w2h2_288[0],
        #                         center_288[0] + w2h2_288[3],
        #                         center_288[1] + w2h2_288[1]]).clamp(min=0, max=287)
        ###############################

        self.debug_info['flag'] = flag
        # print("flag: {}".format(flag))

        # Update position and scale
        if flag != 'not_found':
             self.prev_pos = new_pos.clone()
             if getattr(self.params, 'use_classifier', True):
                 update_scale_flag = getattr(self.params, 'update_scale_when_uncertain', True) or flag == 'normal'
                 pos = (box[:2] + box[2:]) / 2.
                 new_target_sz = box[2:] - box[:2]
                 new_scale = torch.sqrt(new_target_sz.prod() / self.base_target_sz.prod())
                 if update_scale_flag:
                    self.target_scale = new_scale.clamp(self.min_scale_factor, self.max_scale_factor)

                 self.pos = pos.clone()
                 self.target_sz = new_target_sz.clone()


        # ----------------- Online Traing ------------------ #

        update_flag = flag not in ['not_found', 'uncertain']
        hard_negative = (flag == 'hard_negative')
        learning_rate = getattr(self.params, 'hard_negative_learning_rate', None) if hard_negative else None

        if getattr(self.params, 'update_classifier_and_regressor', False) and update_flag:
            # Get train sample
            train_x_72_cls = test_x_72_cls[scale_ind:scale_ind + 1, ...].clone()
            train_x_18_cls = test_x_18_cls[scale_ind:scale_ind + 1, ...].clone()
            train_x_72_reg = test_x_72_reg[scale_ind:scale_ind + 1, ...].clone()

            # Create target_box and label for spatial sample
            target_box = self.get_training_box(self.pos, self.target_sz, sample_pos[scale_ind,:], sample_scales[scale_ind])

            if hard_negative:
                self.update_classifier_72(train_x_72_cls, target_box, learning_rate, s[scale_ind, ...])
                self.update_classifier_18(train_x_18_cls, target_box, learning_rate, s[scale_ind, ...])
                self.update_regressor(train_x_72_reg, target_box, learning_rate, s[scale_ind, ...])
            elif (self.frame_num - 1) <= self.params.init_train_frames:
                self.update_classifier_72(train_x_72_cls, target_box, None, s[scale_ind, ...])
                self.update_classifier_18(train_x_18_cls, target_box, None, s[scale_ind, ...])
                self.update_regressor(train_x_72_reg, target_box, None, s[scale_ind, ...])
            elif (self.frame_num - 1) > self.params.init_train_frames:
                if max_score >= self.max_score or self.train_x_72_cls is None:
                    self.max_score = max_score
                    self.train_x_72_cls = train_x_72_cls
                    self.train_x_72_reg = train_x_72_reg
                    self.target_box = target_box
                    self.train_x_18_cls = train_x_18_cls
                    self.s = s[scale_ind, ...]
                if (self.frame_num - 1) % self.params.train_skipping == 0:
                    if self.params.ues_select_sample_strategy:
                        self.update_classifier_72(self.train_x_72_cls, self.target_box, None, self.s)
                        self.update_classifier_18(self.train_x_18_cls, self.target_box, None, self.s)
                        self.update_regressor(self.train_x_72_reg, self.target_box, None, self.s)
                    else:
                        self.update_classifier_72(train_x_72_cls, target_box, None, s[scale_ind, ...])
                        self.update_classifier_18(train_x_18_cls, target_box, None, s[scale_ind, ...])
                        self.update_regressor(train_x_72_reg, target_box, None, s[scale_ind, ...])
                    self.train_x_72_cls = None
                    self.max_score = -100

        # ----------------- Online Traing End ------------------ #

        score_map = s[scale_ind, ...]
        max_score = torch.max(score_map).item()

        # Compute output bounding box
        new_state = torch.cat((self.pos[[1,0]] - (self.target_sz[[1,0]]-1)/2, self.target_sz[[1,0]]))

        # used to compute the iou between current predict target box and the previous predicted target box.
        self.prev_box = torch.cat((self.pos[[0, 1]] - (self.target_sz[[0, 1]] - 1) / 2, self.pos[[0, 1]] + (self.target_sz[[0, 1]] - 1) / 2))

        # ----------- Visualization ----------- #
        # img_patch = self.im_patches[scale_ind, ...].clone().squeeze().cpu().permute(1, 2, 0).numpy()
        # # print("img_patch shape:{}".format(img_patch.shape))
        # box_draw = box_288.clone().cpu().int().numpy()
        # # print("box_draw: {}".format(box_draw))
        # center = center_288.clone().cpu().int().numpy()
        # center2 = (max_disp2 * 4 + 2).clamp(10, 280).clone().cpu().int().numpy()
        # print("center2: {}".format(center2))
        # print("center: {}".format(center))
        #
        # save_path = "/home/cyt/data/vot-results/"
        # if self.frame_num == 2:
        #     for i in range(1, 1000):
        #         self.seq_path = save_path + str(i)
        #         if not os.path.exists(self.seq_path):
        #             os.makedirs(self.seq_path)
        #             break
        # save_dir = "{}/{}.jpg".format(self.seq_path, self.frame_num)
        # img_draw = self.save_pred_image(img_patch, box_draw, center, center2, save_dir)

        # out = {'target_bbox': new_state.tolist(), 'score_map_18': s[scale_ind, ...], 'score_map_72': score_map,
        #        'image': img_draw}
        # ----------- Visualization End ----------- #

        out = {'target_bbox': new_state.tolist(), 'score_map_18': s[scale_ind, ...], 'score_map_72':score_map,
               'image': self.im_patches[scale_ind, ...].squeeze()}
        return out

    def save_pred_image(self, img, bbox, center, center2, save_file=None):
        """Display a 2D tensor.
        """
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        draw_rec = cv2.rectangle(img, (bbox[1], bbox[0]), (bbox[3], bbox[2]), (0, 0, 255), 2)
        draw_rec = cv2.circle(draw_rec, (center[1], center[0]), 3, (0, 0, 255), -1)
        draw_rec = cv2.circle(draw_rec, (center2[1], center2[0]), 3, (0, 255, 0), -1)
        cv2.imwrite(save_file, draw_rec)
        img = cv2.cvtColor(draw_rec, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img).float().permute(2, 0, 1)
        return img_tensor

    def iou_pred(self, pred, target):
        pred_left = pred[1]
        pred_top = pred[0]
        pred_right = pred[3]
        pred_bottom = pred[2]

        target_left = target[1]
        target_top = target[0]
        target_right = target[3]
        target_bottom = target[2]

        target_area = (target_right - target_left) * \
                      (target_bottom - target_top)
        pred_area = (pred_right - pred_left) * \
                    (pred_bottom - pred_top)

        w_intersect = torch.min(pred_right, target_right) - \
                      torch.max(pred_left, target_left)
        h_intersect = torch.min(pred_bottom, target_bottom) - \
                      torch.max(pred_top, target_top)

        area_intersect = w_intersect * h_intersect
        area_union = target_area + pred_area - area_intersect

        iou = area_intersect / area_union

        return iou.item()


    def ctdet_decode(self, w2h2, max_disp):
        max_disp = max_disp.long()
        w2h2 = w2h2.squeeze()
        w2h2 = (torch.relu(w2h2)) * 4.0
        w2h2_max = w2h2[..., max_disp[0], max_disp[1]]
        return w2h2_max

    def get_sample_location(self, sample_coord):
        """Get the location of the extracted sample."""
        sample_coord = sample_coord.float()
        sample_pos = 0.5*(sample_coord[:,:2] + sample_coord[:,2:] - 1)
        sample_scales = ((sample_coord[:,2:] - sample_coord[:,:2]) / self.img_sample_sz).prod(dim=1).sqrt()
        return sample_pos, sample_scales

    def get_centered_sample_pos(self, feature_sz, kernel_size):
        """Get the center position for the new sample. Make sure the target is correctly centered."""
        return self.pos + ((feature_sz + kernel_size) % 2) * self.target_scale * \
               self.img_support_sz / (2*feature_sz)

    def classify_target_18(self, sample_x: TensorList):
        """Classify target by applying the FCOT classfication filter with size of 18.
        """
        with torch.no_grad():
            scores_cur = self.net.classifier_18.classify(self.target_filter_18, sample_x)
            scores_init = self.net.classifier_18.classify(self.init_target_filter_18, sample_x)
        scores = (1.0 - self.params.lamda_18) * scores_init + self.params.lamda_18 * scores_cur
        return scores

    def classify_target_72(self, sample_x: TensorList):
        """Classify target by applying the FCOT classfication filter with size of 72.
        """
        with torch.no_grad():
            scores_cur = self.net.classifier_72.classify(self.target_filter_72, sample_x)
            scores_init = self.net.classifier_72.classify(self.init_target_filter_72, sample_x)
        scores = (1.0 - self.params.lamda_72) * scores_init + self.params.lamda_72 * scores_cur
        return scores

    def localize_target(self, scores, sample_scales, feature_sz):
        """Run the target localization."""

        scores = scores.squeeze(1)

        if getattr(self.params, 'advanced_localization', False):
            return self.localize_advanced(scores, sample_scales, feature_sz)

        # Get maximum
        score_sz = torch.Tensor(list(scores.shape[-2:]))
        score_center = (score_sz - 1)/2
        max_score, max_disp = dcf.max2d(scores)
        _, scale_ind = torch.max(max_score, dim=0)
        max_disp = max_disp[scale_ind,...].float().cpu().view(-1)
        target_disp = max_disp - score_center

        # Compute translation vector and scale change factor
        translation_vec = target_disp * (self.img_support_sz / feature_sz) * sample_scales[scale_ind]

        return translation_vec, scale_ind, scores, None

    def localize_advanced(self, scores, sample_scales, feature_sz):
        """Run the target advanced localization (as in ATOM).
            w2h2 shape: (1, 1, 4, 72, 72)
        """
        if scores.dim() == 4:
            scores.squeeze(1)

        sz = scores.shape[-2:]
        score_sz = torch.Tensor(list(sz))
        score_center = (score_sz - 1) / 2

        scores_hn = scores
        if self.output_window is not None and getattr(self.params, 'perform_hn_without_windowing', False):
            scores_hn = scores.clone()
            scores *= self.output_window

        max_score1, max_disp1 = dcf.max2d(scores)
        _, scale_ind = torch.max(max_score1, dim=0)
        sample_scale = sample_scales[scale_ind]
        max_score1 = max_score1[scale_ind]
        max_disp1 = max_disp1[scale_ind, ...].float().cpu().view(-1)
        target_disp1 = max_disp1 - score_center
        translation_vec1 = target_disp1 * (self.img_support_sz / feature_sz) * sample_scale

        # print("max_score_{}: {}".format(feature_sz, max_score1.item()))

        # Mask out target neighborhood
        target_neigh_sz = self.params.target_neighborhood_scale * (self.target_sz / sample_scale) * (
                    feature_sz / self.img_support_sz)

        tneigh_top = max(round(max_disp1[0].item() - target_neigh_sz[0].item() / 2), 0)
        tneigh_bottom = min(round(max_disp1[0].item() + target_neigh_sz[0].item() / 2 + 1), sz[0])
        tneigh_left = max(round(max_disp1[1].item() - target_neigh_sz[1].item() / 2), 0)
        tneigh_right = min(round(max_disp1[1].item() + target_neigh_sz[1].item() / 2 + 1), sz[1])
        scores_masked = scores_hn[scale_ind:scale_ind + 1, ...].clone()
        scores_masked[..., tneigh_top:tneigh_bottom, tneigh_left:tneigh_right] = 0

        # Find new maximum
        max_score2, max_disp2 = dcf.max2d(scores_masked)
        # print("max_score2: {}".format(max_score2.item()))

        max_disp2 = max_disp2.float().cpu().view(-1)
        target_disp2 = max_disp2 - score_center
        translation_vec2 = target_disp2 * (self.img_support_sz / feature_sz) * sample_scale

        if max_score1.item() < self.params.target_not_found_threshold:
            return translation_vec1, scale_ind, scores_hn, 'not_found', max_disp1, max_disp2, None

        # Handle the different cases
        if max_score2 > self.params.distractor_threshold * max_score1:
            disp_norm1 = torch.sqrt(torch.sum(target_disp1 ** 2))
            disp_norm2 = torch.sqrt(torch.sum(target_disp2 ** 2))
            disp_threshold = self.params.dispalcement_scale * math.sqrt(sz[0] * sz[1]) / 2

            if disp_norm2 > disp_threshold and disp_norm1 < disp_threshold:
                return translation_vec1, scale_ind, scores_hn, 'hard_negative', max_disp1, max_disp2, max_score1.item()
            if disp_norm2 < disp_threshold and disp_norm1 > disp_threshold:
                return translation_vec2, scale_ind, scores_hn, 'hard_negative', max_disp2, max_disp1, max_score2.item()
            if disp_norm2 > disp_threshold and disp_norm1 > disp_threshold:
                return translation_vec1, scale_ind, scores_hn, 'uncertain', max_disp1, max_disp2, max_score1.item()

            # If also the distractor is close, return with highest score
            return translation_vec1, scale_ind, scores_hn, 'uncertain', max_disp1, max_disp2, max_score1.item()

        if max_score2 > self.params.hard_negative_threshold * max_score1 and max_score2 > self.params.target_not_found_threshold:
            return translation_vec1, scale_ind, scores_hn, 'hard_negative', max_disp1, max_disp2, max_score1.item()

        return translation_vec1, scale_ind, scores_hn, 'normal', max_disp1, max_disp2, max_score1.item()

    def extract_backbone_features(self, im: torch.Tensor, pos: torch.Tensor, scales, sz: torch.Tensor):
        im_patches, patch_coords = sample_patch_multiscale(im, pos, scales, sz, getattr(self.params, 'border_mode', 'replicate'))
        with torch.no_grad():
            backbone_feat = self.net.extract_backbone(im_patches)
        self.im_patches = im_patches
        return backbone_feat, patch_coords

    def get_classification_features(self, backbone_feat):
        with torch.no_grad():
            return self.net.extract_classification_feat_18(backbone_feat)

    def generate_init_samples(self, im: torch.Tensor) -> TensorList:
        """Perform data augmentation to generate initial training samples."""

        if getattr(self.params, 'border_mode', 'replicate') == 'inside':
            # Get new sample size if forced inside the image
            im_sz = torch.Tensor([im.shape[2], im.shape[3]])
            sample_sz = self.target_scale * self.img_sample_sz
            shrink_factor = (sample_sz.float() / im_sz).max().clamp(1)
            sample_sz = (sample_sz.float() / shrink_factor)
            self.init_sample_scale = (sample_sz / self.img_sample_sz).prod().sqrt()
            tl = self.pos - (sample_sz - 1) / 2
            br = self.pos + sample_sz / 2 + 1
            global_shift = - ((-tl).clamp(0) - (br - im_sz).clamp(0)) / self.init_sample_scale
        else:
            self.init_sample_scale = self.target_scale
            global_shift = torch.zeros(2)

        self.init_sample_pos = self.pos.round()

        # Compute augmentation size
        aug_expansion_factor = getattr(self.params, 'augmentation_expansion_factor', None)  # 2
        aug_expansion_sz = self.img_sample_sz.clone()                # [288, 288]
        aug_output_sz = None
        if aug_expansion_factor is not None and aug_expansion_factor != 1:
            aug_expansion_sz = (self.img_sample_sz * aug_expansion_factor).long()
            aug_expansion_sz += (aug_expansion_sz - self.img_sample_sz.long()) % 2
            aug_expansion_sz = aug_expansion_sz.float()
            aug_output_sz = self.img_sample_sz.long().tolist()

        # Random shift for each sample
        get_rand_shift = lambda: None
        random_shift_factor = getattr(self.params, 'random_shift_factor', 0)   # 1/3
        if random_shift_factor > 0:
            get_rand_shift = lambda: ((torch.rand(2) - 0.5) * self.img_sample_sz * random_shift_factor + global_shift).long().tolist()

        # Always put identity transformation first, since it is the unaugmented sample that is always used
        self.transforms = [augmentation.Identity(aug_output_sz, global_shift.long().tolist())]

        augs = self.params.augmentation if getattr(self.params, 'use_augmentation', True) else {}

        # Add all augmentations
        if 'scale' in augs:
            self.transforms.extend([augmentation.Scale(scale_factor, aug_output_sz, get_rand_shift()) for scale_factor in augs['scale']])
        if 'shift' in augs:
            self.transforms.extend([augmentation.Translation(shift, aug_output_sz, global_shift.long().tolist()) for shift in augs['shift']])
        if 'relativeshift' in augs:
            get_absolute = lambda shift: (torch.Tensor(shift) * self.img_sample_sz/2).long().tolist()
            self.transforms.extend([augmentation.Translation(get_absolute(shift), aug_output_sz, global_shift.long().tolist()) for shift in augs['relativeshift']])
        if 'fliplr' in augs and augs['fliplr']:
            self.transforms.append(augmentation.FlipHorizontal(aug_output_sz, get_rand_shift()))
        if 'blur' in augs:
            self.transforms.extend([augmentation.Blur(sigma, aug_output_sz, get_rand_shift()) for sigma in augs['blur']])
        if 'rotate' in augs:
            self.transforms.extend([augmentation.Rotate(angle, aug_output_sz, get_rand_shift()) for angle in augs['rotate']])

        # Extract augmented image patches
        im_patches = sample_patch_transformed(im, self.init_sample_pos, self.init_sample_scale, aug_expansion_sz, self.transforms)

        # Extract initial backbone features
        with torch.no_grad():
            init_backbone_feat = self.net.extract_backbone(im_patches)

        return init_backbone_feat

    def init_target_boxes(self):
        """Get the target bounding boxes for the initial augmented samples."""
        self.classifier_target_box = self.get_training_box(self.pos, self.target_sz, self.init_sample_pos, self.init_sample_scale)
        init_target_boxes = TensorList()
        for T in self.transforms:
            init_target_boxes.append(self.classifier_target_box + torch.Tensor([T.shift[1], T.shift[0], 0, 0]))
        init_target_boxes = torch.cat(init_target_boxes.view(1, 4), 0).to(self.params.device)
        self.target_boxes = init_target_boxes.new_zeros(self.params.sample_memory_size, 4)
        self.target_boxes[:init_target_boxes.shape[0],:] = init_target_boxes
        self.target_boxes_cls_72 = self.target_boxes.clone()
        self.target_boxes_cls_18 = self.target_boxes.clone()
        self.target_boxes_reg_72 = self.target_boxes.clone()
        return init_target_boxes

    def init_memory(self, train_x_72_cls: TensorList,
                    train_x_18_cls: TensorList, train_x_72_reg: TensorList):
        # Initialize first-frame spatial training samples
        self.num_init_samples = train_x_72_cls.size(0)
        init_sample_weights = TensorList([x.new_ones(1) / x.shape[0] for x in train_x_72_cls])

        # Sample counters and weights for spatial
        # self.num_stored_samples = self.num_init_samples.copy()
        # self.previous_replace_ind = [None] * len(self.num_stored_samples)
        # self.sample_weights = TensorList([x.new_zeros(self.params.sample_memory_size) for x in train_x_72_cls])
        # for sw, init_sw, num in zip(self.sample_weights, init_sample_weights, self.num_init_samples):
        #     sw[:num] = init_sw
        # print(self.sample_weights)

        self.num_stored_samples_cls_72 = self.num_init_samples.copy()
        self.num_stored_samples_cls_18 = self.num_init_samples.copy()
        self.num_stored_samples_reg_72 = self.num_init_samples.copy()

        self.previous_replace_ind_cls_72 = [None] * len(self.num_stored_samples_cls_72)
        self.previous_replace_ind_cls_18 = [None] * len(self.num_stored_samples_cls_18)
        self.previous_replace_ind_reg_72 = [None] * len(self.num_stored_samples_reg_72)

        self.sample_weights_cls_72 = TensorList([x.new_zeros(self.params.sample_memory_size) for x in train_x_72_cls])
        for sw, init_sw, num in zip(self.sample_weights_cls_72, init_sample_weights, self.num_init_samples):
            sw[:num] = init_sw

        self.sample_weights_cls_18 = self.sample_weights_cls_72.clone()
        self.sample_weights_reg_72 = self.sample_weights_cls_72.clone()

        # initialize classification features memory
        self.training_samples_cls_72 = TensorList(
            [x.new_zeros(self.params.sample_memory_size, x.shape[1], x.shape[2], x.shape[3]) for x in train_x_72_cls])
        for ts, x in zip(self.training_samples_cls_72, train_x_72_cls):
            ts[:x.shape[0], ...] = x

        self.training_samples_cls_18 = TensorList(
            [x.new_zeros(self.params.sample_memory_size, x.shape[1], x.shape[2], x.shape[3]) for x in train_x_18_cls])
        for ts, x in zip(self.training_samples_cls_18, train_x_18_cls):
            ts[:x.shape[0], ...] = x

        self.training_samples_reg_72 = TensorList(
            [x.new_zeros(self.params.sample_memory_size, x.shape[1], x.shape[2], x.shape[3]) for x in train_x_72_reg])
        for ts, x in zip(self.training_samples_reg_72, train_x_72_reg):
            ts[:x.shape[0], ...] = x


    def update_memory_cls_72(self,sample_x_cls_72: TensorList, target_box, learning_rate = None):
        # Update weights and get replace ind
        replace_ind = self.update_sample_weights(self.sample_weights_cls_72, self.previous_replace_ind_cls_72,
                                                 self.num_stored_samples_cls_72, self.num_init_samples, learning_rate)
        self.previous_replace_ind_cls_72 = replace_ind
        # print("replace: {}".format(replace_ind))
        # print(self.sample_weights)

        # Update sample and label memory
        for train_samp, x, ind in zip(self.training_samples_cls_72, sample_x_cls_72, replace_ind):
            train_samp[ind:ind+1,...] = x

        # Update bb memory
        self.target_boxes_cls_72[replace_ind[0],:] = target_box

        self.num_stored_samples_cls_72 += 1

    def update_memory_cls_18(self,sample_x_cls_18: TensorList, target_box, learning_rate = None):
        # Update weights and get replace ind
        replace_ind = self.update_sample_weights(self.sample_weights_cls_18, self.previous_replace_ind_cls_18,
                                                 self.num_stored_samples_cls_18, self.num_init_samples, learning_rate)
        self.previous_replace_ind_cls_18 = replace_ind
        # print("replace: {}".format(replace_ind))
        # print(self.sample_weights)

        # Update sample and label memory
        for train_samp, x, ind in zip(self.training_samples_cls_18, sample_x_cls_18, replace_ind):
            train_samp[ind:ind+1,...] = x

        # Update bb memory
        self.target_boxes_cls_18[replace_ind[0],:] = target_box

        self.num_stored_samples_cls_18 += 1

    def update_memory_reg_72(self, sample_x_reg_72: TensorList, target_box, learning_rate = None):
        # Update weights and get replace ind
        replace_ind = self.update_sample_weights(self.sample_weights_reg_72, self.previous_replace_ind_reg_72,
                                                 self.num_stored_samples_reg_72, self.num_init_samples, learning_rate)
        self.previous_replace_ind_reg_72 = replace_ind
        # print("replace: {}".format(replace_ind))
        # print(self.sample_weights)

        # Update sample and label memory
        for train_samp, x, ind in zip(self.training_samples_reg_72, sample_x_reg_72, replace_ind):
            train_samp[ind:ind+1,...] = x

        # Update bb memory
        self.target_boxes_reg_72[replace_ind[0],:] = target_box

        self.num_stored_samples_reg_72 += 1

    def update_sample_weights(self, sample_weights, previous_replace_ind, num_stored_samples, num_init_samples, learning_rate = None):
        # Update weights and get index to replace
        replace_ind = []
        for sw, prev_ind, num_samp, num_init in zip(sample_weights, previous_replace_ind, num_stored_samples, num_init_samples):
            lr = learning_rate
            if lr is None:
                lr = self.params.learning_rate

            init_samp_weight = getattr(self.params, 'init_samples_minimum_weight', None)
            if init_samp_weight == 0:
                init_samp_weight = None
            s_ind = 0 if init_samp_weight is None else num_init

            if num_samp == 0 or lr == 1:
                sw[:] = 0
                sw[0] = 1
                r_ind = 0
            else:
                # Get index to replace
                if num_samp < sw.shape[0]:
                    r_ind = num_samp
                else:
                    _, r_ind = torch.min(sw[s_ind:], 0)
                    r_ind = r_ind.item() + s_ind

                # Update weights
                if prev_ind is None:
                    sw /= 1 - lr
                    sw[r_ind] = lr
                else:
                    sw[r_ind] = sw[prev_ind] / (1 - lr)

            sw /= sw.sum()
            if init_samp_weight is not None and sw[:num_init].sum() < init_samp_weight:
                sw /= init_samp_weight + sw[num_init:].sum()
                sw[:num_init] = init_samp_weight / num_init

            replace_ind.append(r_ind)

        return replace_ind

    def get_training_box(self, pos, sz, sample_pos, sample_scale):
        """All inputs in original image coordinates.
        Generates a box in the cropped image sample reference frame, in the format used by the IoUNet."""
        box_center = (pos - sample_pos) / sample_scale + (self.img_sample_sz - 1) / 2
        box_sz = sz / sample_scale
        target_ul = box_center - (box_sz - 1) / 2
        return torch.cat([target_ul.flip((0,)), box_sz.flip((0,))])

    def init_classifier_and_regressor(self, init_backbone_feat):
        # Get classification features
        x = self.net.get_backbone_clf_feat(init_backbone_feat)
        train_feat_18_cls = self.get_classification_features(init_backbone_feat)

        with torch.no_grad():
            train_feat_18 = self.net.pyramid_first_conv(x=None, x_backbone=x)
            train_feat_36 = self.net.pyramid_36(train_feat_18, init_backbone_feat['layer2'])
            train_feat_72 = self.net.pyramid_72(train_feat_36, init_backbone_feat['layer1'])

            train_feat_72_cls = self.net.classifier_72.extract_classification_feat(train_feat_72.
                                                                                   view(-1, *train_feat_72.shape[-3:]))
            train_feat_72_reg = self.net.regressor_72.extract_regression_feat(
                                                            feat_36=train_feat_36.view(-1, *train_feat_36.shape[-3:]),
                                                            feat_72=train_feat_72.view(-1, *train_feat_72.shape[-3:]))

        # Add the dropout augmentation here, since it requires extraction of the classification features
        if 'dropout' in self.params.augmentation and getattr(self.params, 'use_augmentation', True):
            num, prob = self.params.augmentation['dropout']
            self.transforms.extend(self.transforms[:1]*num)
            train_feat_18_cls = torch.cat([train_feat_18_cls,
                                           F.dropout2d(train_feat_18_cls[0:1, ...].
                                                       expand(num, -1, -1, -1), p=prob, training=True)])
            train_feat_72_cls = torch.cat([train_feat_72_cls,
                                           F.dropout2d(train_feat_72_cls[0:1, ...].
                                                       expand(num, -1, -1, -1), p=prob,training=True)])
            train_feat_72_reg = torch.cat([train_feat_72_reg,
                                           F.dropout2d(train_feat_72_reg[0:1, ...].
                                                       expand(num, -1, -1, -1), p=prob,training=True)])

        # Get target boxes for the different augmentations
        target_boxes = self.init_target_boxes()

        # Set number of iterations
        num_iter = getattr(self.params, 'net_opt_iter', None)
        num_iter_72 = getattr(self.params, 'net_opt_iter_72', None)
        reg_num_iter = getattr(self.params, 'reg_net_opt_iter', None)

        # Get target filter by running the discriminative model prediction module
        with torch.no_grad():
            # extract target_filter_72, target_filter_18 and target_reg_filter_72 using Clf and Reg model generators.
            self.target_filter_72, target_filters, losses = self.net.classifier_72.get_filter(train_feat_72_cls,
                                                                                              target_boxes,
                                                                                              num_iter=num_iter_72)
            self.target_filter_18, _, _ = self.net.classifier_18.get_filter(train_feat_18_cls,
                                                                            target_boxes,
                                                                            num_iter=num_iter)

            # get init_reg_filter using target sample and optimize filters using training samples
            target_feat_36 = train_feat_36.view(-1, *train_feat_36.shape[-3:])[0].unsqueeze(0)
            target_feat_72 = train_feat_72.view(-1, *train_feat_72.shape[-3:])[0].unsqueeze(0)
            target_bb = target_boxes[0].unsqueeze(0).clone()
            init_reg_filter = self.net.regressor_72.generate_init_filter(target_feat_36, target_feat_72, target_bb)

            if reg_num_iter > 0:
                self.target_reg_filter_72, _, reg_losses = self.net.regressor_72.generate_filter_optimizer(
                    init_reg_filter, train_feat_72_reg, target_boxes.view(-1, 4).clone(), num_iter=reg_num_iter)
            else:
                self.target_reg_filter_72 = init_reg_filter

            # get initial Clf and Reg model used in tracking process, which merge the initial model and the optimized model.
            self.init_target_filter_72 = self.target_filter_72
            self.init_target_filter_18 = self.target_filter_18
            self.init_reg_filter = init_reg_filter

        # Set feature size and other related sizes
        self.feature_sz_18 = torch.Tensor(list(x.shape[-2:]))
        ksz_18 = self.net.classifier_18.filter_size
        self.kernel_size_18 = torch.Tensor([ksz_18, ksz_18] if isinstance(ksz_18, (int, float)) else ksz_18)
        self.output_sz_18 = self.feature_sz_18 + (self.kernel_size_18 + 1) % 2

        self.feature_sz_72 = torch.Tensor(list(train_feat_72.shape[-2:]))
        ksz_72 = self.net.classifier_72.filter_size
        self.kernel_size_72 = torch.Tensor([ksz_72, ksz_72] if isinstance(ksz_72, (int, float)) else ksz_72)
        self.output_sz_72 = self.feature_sz_72 + (self.kernel_size_72 + 1) % 2
        self.output_sz = torch.Tensor([72, 72])

        # Construct output window
        self.output_window = None
        if getattr(self.params, 'window_output', False):
            if getattr(self.params, 'use_clipped_window', False):
                self.output_window = dcf.hann2d_clipped(
                    self.output_sz.long(),
                    self.output_sz.long() * self.params.effective_search_area / self.params.search_area_scale,
                    centered=False).to(self.params.device)
            else:
                self.output_window = dcf.hann2d(self.output_sz.long(), centered=True).to(self.params.device)
            self.output_window = self.output_window.squeeze(0)

        # Init memory
        if getattr(self.params, 'update_classifier_and_regressor', True):
            self.init_memory(TensorList([train_feat_72_cls]),
                             TensorList([train_feat_18_cls]), TensorList([train_feat_72_reg]))

    def update_classifier_72(self, train_x_cls_72, target_box, learning_rate=None, scores=None):
        # Set flags and learning rate
        hard_negative_flag = learning_rate is not None
        if learning_rate is None:
            learning_rate = self.params.learning_rate

        # Update the tracker memory
        self.update_memory_cls_72(TensorList([train_x_cls_72]), target_box, learning_rate)

        # Decide the number of iterations to run
        num_iter = 0
        low_score_th = getattr(self.params, 'low_score_opt_threshold', None)
        if hard_negative_flag:
            num_iter = getattr(self.params, 'net_opt_hn_iter_72', None)
        elif low_score_th is not None and low_score_th > scores.max().item():
            num_iter = getattr(self.params, 'net_opt_low_iter', None)
        elif (self.frame_num - 1) <= self.params.init_train_frames:
            num_iter = self.params.init_train_iter_72
        elif (self.frame_num - 1) % self.params.train_skipping == 0:
            num_iter = getattr(self.params, 'net_opt_update_iter_72', None)

        if num_iter > 0:
            # Get inputs for the DiMP filter optimizer module
            samples_72_cls = self.training_samples_cls_72[0][:self.num_stored_samples_cls_72[0],...]
            target_boxes = self.target_boxes_cls_72[:self.num_stored_samples_cls_72[0],:].clone()
            sample_weights = self.sample_weights_cls_72[0][:self.num_stored_samples_cls_72[0]]

            # Run the model optimizer module
            with torch.no_grad():
                self.target_filter_72, _, losses = self.net.classifier_72.filter_optimizer(self.target_filter_72,
                                                                                           samples_72_cls,
                                                                                           target_boxes,
                                                                                           sample_weight=sample_weights,
                                                                                           num_iter=num_iter)


    def update_classifier_18(self, train_x_cls_18, target_box, learning_rate=None, scores=None):
        # Set flags and learning rate
        hard_negative_flag = learning_rate is not None
        if learning_rate is None:
            learning_rate = self.params.learning_rate

        # Update the tracker memory
        self.update_memory_cls_18(TensorList([train_x_cls_18]), target_box, learning_rate)

        # Decide the number of iterations to run
        num_iter = 0
        low_score_th = getattr(self.params, 'low_score_opt_threshold', None)
        if hard_negative_flag:
            num_iter = getattr(self.params, 'net_opt_hn_iter', None)
        elif low_score_th is not None and low_score_th > scores.max().item():
            num_iter = getattr(self.params, 'net_opt_low_iter', None)
        elif (self.frame_num - 1) <= self.params.init_train_frames:
            num_iter = self.params.init_train_iter
        elif (self.frame_num - 1) % self.params.train_skipping == 0:
            num_iter = getattr(self.params, 'net_opt_update_iter', None)

        if num_iter > 0:
            # Get inputs for the DiMP filter optimizer module
            samples_18_cls = self.training_samples_cls_18[0][:self.num_stored_samples_cls_18[0], ...]
            target_boxes = self.target_boxes_cls_18[:self.num_stored_samples_cls_18[0],:].clone()
            sample_weights = self.sample_weights_cls_18[0][:self.num_stored_samples_cls_18[0]]

            # Run the model optimizer module
            with torch.no_grad():
                self.target_filter_18, _, losses = self.net.classifier_18.filter_optimizer(self.target_filter_18,
                                                                                           samples_18_cls,
                                                                                           target_boxes,
                                                                                           sample_weight=sample_weights,
                                                                                           num_iter=num_iter)


    def update_regressor(self, train_x_reg_72, target_box, learning_rate=None, scores=None):
        # Set flags and learning rate
        hard_negative_flag = learning_rate is not None
        if learning_rate is None:
            learning_rate = self.params.learning_rate

        # Update the tracker memory
        self.update_memory_reg_72(TensorList([train_x_reg_72]), target_box, learning_rate)

        # Decide the number of iterations to run
        reg_num_iter = 0
        low_score_th = getattr(self.params, 'low_score_opt_threshold', None)
        if hard_negative_flag:
            reg_num_iter = getattr(self.params, 'reg_net_opt_hn_iter', None)
        elif low_score_th is not None and low_score_th > scores.max().item():
            reg_num_iter = 0
        elif (self.frame_num - 1) <= self.params.init_train_frames:
            reg_num_iter = self.params.reg_init_train_iter
        elif (self.frame_num - 1) % self.params.train_skipping == 0:
            reg_num_iter = getattr(self.params, 'reg_net_opt_update_iter', None)

        if reg_num_iter > 0:
            # Get inputs for the DiMP filter optimizer module
            samples_72_reg = self.training_samples_reg_72[0][:self.num_stored_samples_reg_72[0], ...]
            target_boxes = self.target_boxes_reg_72[:self.num_stored_samples_reg_72[0],:].clone()
            sample_weights = self.sample_weights_reg_72[0][:self.num_stored_samples_reg_72[0]]

            # Run the model optimizer module
            with torch.no_grad():
                self.target_reg_filter_72, _, reg_losses = self.net.regressor_72.generate_filter_optimizer(init_filter=self.target_reg_filter_72,
                                                                                                           feat=samples_72_reg,
                                                                                                           bb=target_boxes,
                                                                                                           num_iter=reg_num_iter)