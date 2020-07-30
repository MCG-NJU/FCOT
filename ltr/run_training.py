import os
import sys
import argparse
import importlib
import multiprocessing
import cv2 as cv
import torch.backends.cudnn

env_path = os.path.join(os.path.dirname(__file__), '..')
if env_path not in sys.path:
    sys.path.append(env_path)

import ltr.admin.settings as ws_settings


def run_training(train_module, train_name, cudnn_benchmark=True, args=None):
    """Run a train scripts in train_settings.
    args:
        train_module: Name of module in the "train_settings/" folder.
        train_name: Name of the train settings file.
        cudnn_benchmark: Use cudnn benchmark or not (default is True).
    """

    # This is needed to avoid strange crashes related to opencv
    cv.setNumThreads(0)

    torch.backends.cudnn.benchmark = cudnn_benchmark

    print('Training:  {}  {}'.format(train_module, train_name))

    settings = ws_settings.Settings()
    settings.module_name = train_module
    settings.script_name = train_name
    settings.project_path = 'ltr/{}/{}'.format(train_module, train_name)

    settings.samples_per_epoch = args.samples_per_epoch
    settings.use_pretrained_dimp = args.use_pretrained_dimp
    settings.pretrained_dimp50 = args.pretrained_dimp50
    settings.load_model = args.load_model
    settings.fcot_model = args.fcot_model
    settings.train_cls_72_and_reg_init = args.train_cls_72_and_reg_init
    settings.train_reg_optimizer = args.train_reg_optimizer
    settings.train_cls_18 = args.train_cls_18
    settings.total_epochs = args.total_epochs
    settings.lasot_rate = args.lasot_rate
    settings.devices_id = args.devices_id
    settings.batch_size = args.batch_size
    settings.num_workers = args.num_workers
    settings.norm_scale_coef = args.norm_scale_coef

    if args.workspace_dir is not None:
        settings.env.workspace_dir = args.workspace_dir
        settings.env.tensorboard_dir = settings.env.workspace_dir + '/tensorboard/'

    expr_module = importlib.import_module('ltr.train_settings.{}.{}'.format(train_module, train_name))
    expr_func = getattr(expr_module, 'run')

    expr_func(settings)

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

def main():
    parser = argparse.ArgumentParser(description='Run a train scripts in train_settings.')
    parser.add_argument('train_module', type=str, help='Name of module in the "train_settings/" folder.')
    parser.add_argument('train_name', type=str, help='Name of the train settings file.')
    parser.add_argument('--cudnn_benchmark', type=bool, default=True, help='Set cudnn benchmark on (1) or off (0) (default is on).')

    parser.add_argument('--samples_per_epoch', type=int, default=26000)
    parser.add_argument('--use_pretrained_dimp', type=boolean_string, default=False)
    parser.add_argument('--pretrained_dimp50', type=str, default=None)
    parser.add_argument('--load_model', type=boolean_string, default=False)
    parser.add_argument('--fcot_model', type=str, default=None)
    parser.add_argument('--train_cls_72_and_reg_init', type=boolean_string, default=False)
    parser.add_argument('--train_reg_optimizer', type=boolean_string, default=False)
    parser.add_argument('--train_cls_18', type=boolean_string, default=False)
    parser.add_argument('--total_epochs', type=int, default=80)
    parser.add_argument('--workspace_dir', type=str, default=None)
    parser.add_argument('--lasot_rate', type=float, default=0.25)
    parser.add_argument('--norm_scale_coef', type=float, default=2.0)
    parser.add_argument('--batch_size', type=int, default=40)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--devices_id', type=int, nargs='+', default=[0,1,2,3,4,5,6,7], help="Gpus used for training.")


    args = parser.parse_args()

    run_training(args.train_module, args.train_name, args.cudnn_benchmark, args)


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    main()
