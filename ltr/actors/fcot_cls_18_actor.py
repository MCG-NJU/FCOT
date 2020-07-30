from . import BaseActor
from ltr.models.loss.target_regression import DenseIouPred

class FcotCls18Actor(BaseActor):
    """Actor for training the FCOT network."""
    def __init__(self, net, objective, device, loss_weight=None):
        super().__init__(net, objective, device)
        if loss_weight is None:
            loss_weight = {'test_clf_72': 0, 'test_init_clf_72': 0, 'test_iter_clf_72': 0,
                           'test_clf_18': 100, 'test_init_clf_18': 100, 'test_iter_clf_18': 400,
                           'reg_72': 0}
        self.loss_weight = loss_weight
        self.dense_iou_pred = DenseIouPred(dim=4)

    def __call__(self, data, gen_iou_map=False):
        """
        args:
            data - The input data.
            gen_iou_map -

        returns:
            loss  - the training loss
            stats  -  dict containing detailed losses
        """
        # Run network
        target_scores_18, target_scores_72, offset_maps = self.net(train_imgs=data['train_images'],
                                                                   test_imgs=data['test_images'],
                                                                   train_bb=data['train_anno'])

        if gen_iou_map:
            pred_iou_map = self.dense_iou_pred(offset_maps[-1], data['ind_72'], data['w2h2_72'], radius=5)     # [h, w]
        else:
            pred_iou_map = None


        ### Classification-18 losses ###
        clf_losses_test_18 = [self.objective['test_clf_18'](s, data['test_label'], data['test_anno']) for s in
                              target_scores_18]

        # Loss of the final filter
        clf_loss_test_18 = clf_losses_test_18[-1]
        loss_target_classifier_18 = self.loss_weight['test_clf_18'] * clf_loss_test_18

        # Loss for the initial filter iteration
        loss_test_init_clf_18 = 0
        if 'test_init_clf_18' in self.loss_weight.keys():
            loss_test_init_clf_18 = self.loss_weight['test_init_clf_18'] * clf_losses_test_18[0]

        # Loss for the intermediate filter iterations
        loss_test_iter_clf_18 = 0
        if 'test_iter_clf_18' in self.loss_weight.keys():
            test_iter_weights_18 = self.loss_weight['test_iter_clf_18']
            if isinstance(test_iter_weights_18, list):
                loss_test_iter_clf_18 = sum(
                    [a * b for a, b in zip(test_iter_weights_18, clf_losses_test_18[1:-1])])
            else:
                loss_test_iter_clf_18 = (test_iter_weights_18 / (len(clf_losses_test_18) - 2)) * sum(
                    clf_losses_test_18[1:-1])

        # Total loss
        loss = loss_target_classifier_18 + loss_test_init_clf_18 + loss_test_iter_clf_18

        # Log stats
        stats = {'Loss/total': loss.item(),
                 'Loss/target_clf_18': loss_target_classifier_18.item(),
                }

        if 'test_init_clf_18' in self.loss_weight.keys():
            stats['Loss/test_init_clf_18'] = loss_test_init_clf_18.item()
        if 'test_iter_clf_18' in self.loss_weight.keys():
            stats['Loss/test_iter_clf_18'] = loss_test_iter_clf_18.item()

        return loss, stats, target_scores_72[-1][:,0, ...], data['test_images'][:,0, ...], pred_iou_map

