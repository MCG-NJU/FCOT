import os
from collections import OrderedDict
from ltr.trainers import BaseTrainer
from ltr.admin.stats import AverageMeter, StatValue
from ltr.admin.tensorboard import TensorboardWriter
import torch
import time
import cv2 as cv
import numpy as np
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import logging

class LTRFcotTrainer(BaseTrainer):
    def __init__(self, actor, loaders, optimizer, settings, basic_device, lr_scheduler=None, logging_file="log.txt"):
        """
        args:
            actor - The actor for training the network
            loaders - list of dataset loaders, e.g. [train_loader, val_loader]. In each epoch, the trainer runs one
                        epoch for each loader.
            optimizer - The optimizer used for training, e.g. Adam
            settings - Training settings
            lr_scheduler - Learning rate scheduler
            logging_file - The file for logging the losses.
        """
        super().__init__(actor, loaders, optimizer, settings, basic_device, lr_scheduler, logging_file)

        self._set_default_settings()

        # Initialize statistics variables
        self.stats = OrderedDict({loader.name: None for loader in self.loaders})

        # Initialize tensorboard
        tensorboard_writer_dir = os.path.join(self.settings.env.tensorboard_dir, self.settings.project_path)
        self.tensorboard_writer = TensorboardWriter(tensorboard_writer_dir, [l.name for l in loaders])
        self.tensorboard_writer_dir = tensorboard_writer_dir

        self.eval_writer = SummaryWriter(os.path.join(self.tensorboard_writer_dir, 'eval_otb'))
        self.train_image_writer = SummaryWriter(os.path.join(self.tensorboard_writer_dir, 'train_vis'))
        self.val_image_writer = SummaryWriter(os.path.join(self.tensorboard_writer_dir, 'val_vis'))

        self.train_count = 0
        self.val_count = 0

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(level=logging.INFO)
        handler = logging.FileHandler(logging_file)
        formatter = logging.Formatter('%(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def _set_default_settings(self):
        # Dict of all default values
        default = {'print_interval': 10,
                   'print_stats': None,
                   'description': ''}

        for param, default_value in default.items():
            if getattr(self.settings, param, None) is None:
                setattr(self.settings, param, default_value)

    def cycle_dataset(self, loader):
        """Do a cycle of training or validation."""
        self.actor.train(loader.training)
        torch.set_grad_enabled(loader.training)
        self._init_timing()

        for i, data in enumerate(loader, 1):
            # get inputs
            data = data.to(self.device)
            data['epoch'] = self.epoch
            data['settings'] = self.settings

            # forward pass
            if i % 100 == 0:
                loss, stats, scores_72, test_images, pred_iou_map = self.actor(data, gen_iou_map=True)

                images = make_grid(test_images, nrow=4, padding=2, normalize=True, scale_each=False, pad_value=0)
                scores_72 = self.tensors_gray2rgb(scores_72.unsqueeze(1))
                scores_72 = make_grid(scores_72, nrow=4, padding=2, normalize=True, scale_each=False, pad_value=0)
                pred_iou_map = self.tensors_gray2rgb(pred_iou_map.unsqueeze(0).unsqueeze(1))
                pred_iou_map = make_grid(pred_iou_map, nrow=1, padding=2, normalize=False, scale_each=False, pad_value=0)
                print(loader.name)
                if loader.name == 'train':
                    self.train_image_writer.add_image('test_images', images, global_step=self.train_count)
                    self.train_image_writer.add_image('scores_72', scores_72, global_step=self.train_count)
                    self.train_image_writer.add_image('pred_iou_map', pred_iou_map, global_step=self.train_count)
                    self.train_count += 1
                elif loader.name == 'val':
                    self.val_image_writer.add_image('test_images', images, global_step=self.val_count)
                    self.val_image_writer.add_image('scores_72', scores_72, global_step=self.val_count)
                    self.val_image_writer.add_image('pred_iou_map', pred_iou_map, global_step=self.val_count)
                    self.val_count += 1
                print("Displayed heat maps in tensorboard done.")
            else:
                loss, stats, scores_72, test_images, pred_iou_map = self.actor(data, gen_iou_map=False)

            # backward pass and update weights
            if loader.training:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # update statistics
            batch_size = data['train_images'].shape[loader.stack_dim]
            self._update_stats(stats, batch_size, loader)

            # print statistics
            self._print_stats(i, loader, batch_size)

    def train_epoch(self):
        """Do one epoch for each loader."""
        for loader in self.loaders:
            if self.epoch % loader.epoch_interval == 0:
                self.cycle_dataset(loader)

        self._stats_new_epoch()
        self._write_tensorboard()

    def _init_timing(self):
        self.num_frames = 0
        self.start_time = time.time()
        self.prev_time = self.start_time

    def _update_stats(self, new_stats: OrderedDict, batch_size, loader):
        # Initialize stats if not initialized yet
        if loader.name not in self.stats.keys() or self.stats[loader.name] is None:
            self.stats[loader.name] = OrderedDict({name: AverageMeter() for name in new_stats.keys()})

        for name, val in new_stats.items():
            if name not in self.stats[loader.name].keys():
                self.stats[loader.name][name] = AverageMeter()
            self.stats[loader.name][name].update(val, batch_size)

    def _print_stats(self, i, loader, batch_size):
        self.num_frames += batch_size
        current_time = time.time()
        batch_fps = batch_size / (current_time - self.prev_time)
        average_fps = self.num_frames / (current_time - self.start_time)
        self.prev_time = current_time
        if i % self.settings.print_interval == 0 or i == loader.__len__():
            print_str = '[%s: %d, %d / %d] ' % (loader.name, self.epoch, i, loader.__len__())
            print_str += 'FPS: %.1f (%.1f)  ,  ' % (average_fps, batch_fps)
            for name, val in self.stats[loader.name].items():
                if (self.settings.print_stats is None or name in self.settings.print_stats) and hasattr(val, 'avg'):
                    print_str += '%s: %.5f  ,  ' % (name, val.avg)
            print(print_str[:-5])
            self.logger.info(print_str[:-5])


    def _stats_new_epoch(self):
        # Record learning rate
        for loader in self.loaders:
            if loader.training:
                lr_list = self.lr_scheduler.get_lr()
                for i, lr in enumerate(lr_list):
                    var_name = 'LearningRate/group{}'.format(i)
                    if var_name not in self.stats[loader.name].keys():
                        self.stats[loader.name][var_name] = StatValue()
                    self.stats[loader.name][var_name].update(lr)

        for loader_stats in self.stats.values():
            if loader_stats is None:
                continue
            for stat_value in loader_stats.values():
                if hasattr(stat_value, 'new_epoch'):
                    stat_value.new_epoch()

    def _write_tensorboard(self):
        if self.epoch == 1:
            self.tensorboard_writer.write_info(self.settings.module_name, self.settings.script_name, self.settings.description)

        self.tensorboard_writer.write_epoch(self.stats, self.epoch)

    def tensors_gray2rgb(self, tensors):
        """Convert to rgb images [tenosrs： (num, 1, h, w)].
           :return tensors： (num, 3, h, w)
        """

        tmp = os.path.join(self.settings.env.workspace_dir, 'temp.png')
        out = []
        for tensor in tensors:
            image = tensor.clone().squeeze().cpu().detach().numpy()  # [h, w]
            plt.imsave(tmp, image)
            rgb = plt.imread(tmp)[..., 0:3]      # [h, w, 3]
            rgb = torch.from_numpy(rgb).permute(2, 0, 1)
            out.append(rgb)
            # print(rgb.shape)
        out = torch.stack(out, dim=0).float().to(tensors.device)

        return out


    def save_tensor_as_image(self, a: torch.Tensor, save_file = None, range=(None, None)):
        """Display a 2D tensor.
        args:
            fig_num: Figure number.
            title: Title of figure.
        """
        a_np = a.squeeze()
        if a_np.ndim == 3:
            a_np = np.transpose(a_np, (1, 2, 0))
        plt.imsave(save_file, a_np)


    def _read_image(self, image_file: str):
        return cv.cvtColor(cv.imread(image_file), cv.COLOR_BGR2RGB)