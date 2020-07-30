from pytracking.utils import TrackerParams
from pytracking.features.net_wrappers import NetWithBackbone

def parameters():
    params = TrackerParams()

    params.debug = 0
    params.visualization = False

    params.use_gpu = True
    params.use_classifier = True
    params.image_sample_size = 18 * 16
    params.search_area_scale = 4.5

    # params.scale_factors = [0.975, 1.0, 1.025]

    ### Learning parameters
    params.sample_memory_size = 50
    params.learning_rate = 0.01
    params.init_samples_minimum_weight = 0.25
    params.train_skipping = 14
    params.init_train_frames = 10

    ### Net optimization params
    params.update_classifier_and_regressor = True
    params.ues_select_sample_strategy = True

    # classifier-18
    params.init_train_iter = 6
    params.net_opt_iter = 5
    params.net_opt_update_iter = 2
    params.net_opt_hn_iter = 1

    # classifier-72
    params.init_train_iter_72 = 6
    params.net_opt_iter_72 = 5
    params.net_opt_update_iter_72 = 2
    params.net_opt_hn_iter_72 = 1

    # regressor
    params.reg_init_train_iter = 6
    params.reg_net_opt_iter = 4
    params.reg_net_opt_hn_iter = 0
    params.reg_net_opt_update_iter = 1

    ### merge the initial model and the optimized model.
    params.lamda_72 = 1
    params.lamda_18 = 1
    params.reg_lamda = 1

    ### multi-scale classification
    params.merge_rate_72 = 0.4
    params.merge_rate_18 = 0.6

    ### Init augmentation parameters
    params.use_augmentation = True
    params.augmentation = {'fliplr': True,
                           'rotate': [5, -5, 10, -10, 20, -20, 30, -30, 45, -45, -60, 60],
                           'blur': [(2, 0.2), (0.2, 2), (3, 1), (1, 3), (2, 2)],
                           'relativeshift': [(0.6, 0.6), (-0.6, 0.6), (0.6, -0.6), (-0.6, -0.6)],
                           # 'dropout': (7, 0.2)
                           }

    params.augmentation_expansion_factor = 2
    params.random_shift_factor = 1/3

    ### Advanced localization parameters
    params.advanced_localization = True
    params.target_not_found_threshold = 0
    params.distractor_threshold = 100
    params.hard_negative_threshold = 0.45
    params.target_neighborhood_scale = 2.2
    params.dispalcement_scale = 0.7

    params.window_output = True
    params.perform_hn_without_windowing = True
    params.hard_negative_learning_rate = 0.02
    params.update_scale_when_uncertain = True

    params.iou_select = False

    params.net = NetWithBackbone(net_path='fcot.pth',
                                 use_gpu=params.use_gpu)
    params.net.initialize()

    params.vot_anno_conversion_type = 'preserve_area'

    return params
