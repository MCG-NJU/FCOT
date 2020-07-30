import torch.optim as optim
import torchvision.transforms
import torch
from ltr.dataset import Lasot, Got10k, TrackingNet, MSCOCOSeq
from ltr.data import sampler, LTRLoader, processing_fcot
from ltr.models.tracking import fcotnet
import ltr.models.loss as ltr_losses
from ltr.models.loss.target_regression import REGLoss
from ltr import actors
from ltr.trainers import LTRFcotTrainer
import ltr.data.transforms as dltransforms
from ltr import MultiGPU
import collections

# Address the issue: "RuntimeError: received 0 items of ancdata"
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))
torch.multiprocessing.set_sharing_strategy('file_system')


def run(settings):
    settings.description = 'Default train settings for FCOT with ResNet50 as backbone.'
    settings.multi_gpu = True
    settings.print_interval = 1
    settings.normalize_mean = [0.485, 0.456, 0.406]
    settings.normalize_std = [0.229, 0.224, 0.225]
    settings.search_area_factor = 5.0
    settings.output_sigma_factor = 1/4
    settings.clf_target_filter_sz = 4
    settings.reg_target_filter_sz = 3
    settings.feature_sz = 18
    settings.output_sz = settings.feature_sz * 16
    settings.center_jitter_factor = {'train': 3, 'test': 4.5}
    settings.scale_jitter_factor = {'train': 0.25, 'test': 0.5}
    settings.hinge_threshold = 0.05
    settings.logging_file = 'fcot_log.txt'

    # Train datasets
    lasot_train = Lasot(settings.env.lasot_dir, split='train')
    got10k_train = Got10k(settings.env.got10k_dir, split='vottrain')
    trackingnet_train = TrackingNet(settings.env.trackingnet_dir, set_ids=list(range(4)))
    coco_train = MSCOCOSeq(settings.env.coco_dir)

    # Validation datasets
    got10k_val = Got10k(settings.env.got10k_dir, split='votval')


    # Data transform
    transform_joint = dltransforms.ToGrayscale(probability=0.05)

    transform_train = torchvision.transforms.Compose([dltransforms.ToTensorAndJitter(0.2),
                                                      torchvision.transforms.Normalize(mean=settings.normalize_mean, std=settings.normalize_std)])

    transform_val = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Normalize(mean=settings.normalize_mean, std=settings.normalize_std)])

    # The tracking pairs processing module
    output_sigma = settings.output_sigma_factor / settings.search_area_factor
    proposal_params = {'min_iou': 0.1, 'boxes_per_frame': 8, 'sigma_factor': [0.01, 0.05, 0.1, 0.2, 0.3]}
    label_params = {'feature_sz': settings.feature_sz, 'sigma_factor': output_sigma, 'kernel_sz': settings.clf_target_filter_sz}
    data_processing_train = processing_fcot.AnchorFreeProcessing(search_area_factor=settings.search_area_factor,
                                                                 output_sz=settings.output_sz,
                                                                 center_jitter_factor=settings.center_jitter_factor,
                                                                 scale_jitter_factor=settings.scale_jitter_factor,
                                                                 mode='sequence',
                                                                 output_spatial_scale=72 / 288.,
                                                                 proposal_params=proposal_params,
                                                                 label_function_params=label_params,
                                                                 transform=transform_train,
                                                                 joint_transform=transform_joint)

    data_processing_val = processing_fcot.AnchorFreeProcessing(search_area_factor=settings.search_area_factor,
                                                               output_sz=settings.output_sz,
                                                               center_jitter_factor=settings.center_jitter_factor,
                                                               scale_jitter_factor=settings.scale_jitter_factor,
                                                               mode='sequence',
                                                               output_spatial_scale=72 / 288.,
                                                               proposal_params=proposal_params,
                                                               label_function_params=label_params,
                                                               transform=transform_val,
                                                               joint_transform=transform_joint)

    # Train sampler and loader
    dataset_train = sampler.FCOTSampler([lasot_train, got10k_train, trackingnet_train, coco_train], [settings.lasot_rate,1,1,1],
                                        samples_per_epoch=settings.samples_per_epoch, max_gap=30, num_test_frames=3, num_train_frames=3,
                                        processing=data_processing_train)

    loader_train = LTRLoader('train', dataset_train, training=True, batch_size=settings.batch_size,
                             num_workers=settings.num_workers, shuffle=True, drop_last=True, stack_dim=1)

    # Validation samplers and loaders
    dataset_val = sampler.FCOTSampler([got10k_val], [1], samples_per_epoch=5000, max_gap=30, num_test_frames=3,
                                      num_train_frames=3, processing=data_processing_val)

    loader_val = LTRLoader('val', dataset_val, training=False, batch_size=settings.batch_size, epoch_interval=5,
                           num_workers=settings.num_workers, shuffle=False, drop_last=True, stack_dim=1)

    # Create network
    net = fcotnet.fcotnet(clf_filter_size=settings.clf_target_filter_sz, reg_filter_size=settings.reg_target_filter_sz,
                          backbone_pretrained=True, optim_iter=5, norm_scale_coef=settings.norm_scale_coef,
                          clf_feat_norm=True, clf_feat_blocks=0, final_conv=True, out_feature_dim=512,
                          optim_init_step=0.9, optim_init_reg=0.1, init_gauss_sigma=output_sigma * settings.feature_sz,
                          num_dist_bins=100, bin_displacement=0.1, mask_init_factor=3.0, target_mask_act='sigmoid',
                          score_act='relu', train_reg_optimizer=settings.train_reg_optimizer,
                          train_cls_72_and_reg_init=settings.train_cls_72_and_reg_init, train_cls_18=settings.train_cls_18)

    # Load dimp-model as initial weights
    device = torch.device('cuda:{}'.format(settings.devices_id[0]) if torch.cuda.is_available() else 'cpu')
    if settings.use_pretrained_dimp:
        assert settings.pretrained_dimp50 is not None
        dimp50 = torch.load(settings.pretrained_dimp50, map_location=device)
        state_dict = collections.OrderedDict()
        for key, v in dimp50['net'].items():
            if key.split('.')[0] == 'feature_extractor':
                state_dict['.'.join(key.split('.')[1:])] = v

        net.feature_extractor.load_state_dict(state_dict)

        state_dict = collections.OrderedDict()
        for key, v in dimp50['net'].items():
            if key.split('.')[0] == 'classifier':
                state_dict['.'.join(key.split('.')[1:])] = v
        net.classifier_18.load_state_dict(state_dict)
        print("loading backbone and Classifier modules from DiMP50 done.")

    # Load fcot-model trained in the previous stage
    if settings.load_model:
        assert settings.fcot_model is not None
        load_dict = torch.load(settings.fcot_model)
        fcot_dict = net.state_dict()
        load_fcotnet_dict = {k: v for k, v in load_dict['net'].items() if k in fcot_dict}
        fcot_dict.update(load_fcotnet_dict)
        net.load_state_dict(fcot_dict)
        print("loading FCOT model done.")

    # Wrap the network for multi GPU training
    if settings.multi_gpu:
        net = MultiGPU(net, device_ids=settings.devices_id, dim=1).to(device)

    # Loss for cls_72, cls_18 and regression
    objective = {'test_clf_72': ltr_losses.LBHinge(threshold=settings.hinge_threshold),
                 'test_clf_18': ltr_losses.LBHinge(threshold=settings.hinge_threshold),
                 'reg_72': REGLoss(dim=4)
                 }

    # Create actor and adam-optimizer
    if settings.train_cls_72_and_reg_init and settings.train_cls_18:
        ### train regression branch and clssification branches jointly, except for regression optimizer (TODO: fix)
        print("train cls_72, cls_18 and reg_init jointly...")
        loss_weight = {'test_clf_72': 100, 'test_init_clf_72': 100, 'test_iter_clf_72': 400,
                       'test_clf_18': 100, 'test_init_clf_18': 100, 'test_iter_clf_18': 400,
                       'reg_72': 1}
        actor = actors.FcotActor(net=net, objective=objective, loss_weight=loss_weight, device=device)
        optimizer = optim.Adam([{'params': actor.net.classifier_72.filter_initializer.parameters(), 'lr': 5e-5},
                                {'params': actor.net.classifier_72.filter_optimizer.parameters(), 'lr': 5e-4},
                                {'params': actor.net.classifier_72.feature_extractor.parameters(), 'lr': 5e-5},
                                {'params': actor.net.classifier_18.filter_initializer.parameters(), 'lr': 5e-5},
                                {'params': actor.net.classifier_18.filter_optimizer.parameters(), 'lr': 5e-4},
                                {'params': actor.net.classifier_18.feature_extractor.parameters(), 'lr': 5e-5},
                                {'params': actor.net.regressor_72.parameters()},
                                {'params': actor.net.pyramid_first_conv.parameters()},
                                {'params': actor.net.pyramid_36.parameters()},
                                {'params': actor.net.pyramid_72.parameters()},
                                {'params': actor.net.feature_extractor.parameters(), 'lr': 2e-5}],
                                lr=2e-4)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[35, 46, 60], gamma=0.2)
    elif settings.train_cls_72_and_reg_init:
        # Setting of the first training stage: train backbone, cls_72 and regression (except for regression optimizer) branch.
        print("train cls_72 and reg_init...")
        loss_weight = {'test_clf_72': 100, 'test_init_clf_72': 10, 'test_iter_clf_72': 400,
                       'test_clf_18': 0, 'test_init_clf_18': 0, 'test_iter_clf_18': 0,
                       'reg_72': 0.3}
        actor = actors.FcotCls72AndRegInitActor(net=net, objective=objective, loss_weight=loss_weight, device=device)
        optimizer = optim.Adam([{'params': actor.net.classifier_72.filter_initializer.parameters(), 'lr': 5e-5},
                                {'params': actor.net.classifier_72.filter_optimizer.parameters(), 'lr': 5e-4},
                                {'params': actor.net.classifier_72.feature_extractor.parameters(), 'lr': 5e-5},
                                {'params': actor.net.regressor_72.parameters()},
                                {'params': actor.net.pyramid_first_conv.parameters()},
                                {'params': actor.net.pyramid_36.parameters()},
                                {'params': actor.net.pyramid_72.parameters()},
                                {'params': actor.net.feature_extractor.parameters(), 'lr': 2e-5}],
                                lr=2e-4)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[35, 45, 69], gamma=0.2)
    elif settings.train_reg_optimizer:
        # Setting of the second training stage: train regression optimizer.
        print("train regression optimizer...")
        loss_weight = {'test_reg_72': 1, 'test_init_reg_72': 0, 'test_iter_reg_72': 1}
        actor = actors.FcotOnlineRegressionActor(net=net, objective=objective, loss_weight=loss_weight, device=device)
        optimizer = optim.Adam([{'params': actor.net.regressor_72.filter_optimizer.parameters()}],
                               lr=5e-4)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2], gamma=0.2)
    elif settings.train_cls_18:
        print("train cls_18...")
        # Setting of the third training stage: train cls_18 branch.
        loss_weight = {'test_clf_18': 100, 'test_init_clf_18': 100, 'test_iter_clf_18': 400}
        actor = actors.FcotCls18Actor(net=net, objective=objective, loss_weight=loss_weight, device=device)
        optimizer = optim.Adam([{'params': actor.net.classifier_18.filter_initializer.parameters(), 'lr': 5e-5},
                                {'params': actor.net.classifier_18.filter_optimizer.parameters(), 'lr': 5e-4},
                                {'params': actor.net.classifier_18.feature_extractor.parameters(), 'lr': 5e-5}],
                                lr=2e-4)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[25], gamma=0.2)
    else:
        # TODO: train jointly
        raise Exception("Please run training in correct way.")

    trainer = LTRFcotTrainer(actor, [loader_train, loader_val], optimizer, settings, device, lr_scheduler,
                             logging_file=settings.logging_file)

    trainer.train(settings.total_epochs, load_latest=True, fail_safe=True)