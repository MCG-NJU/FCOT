from . import BaseActor
from ltr.models.loss.target_regression import DenseIouPred
from ltr.models.loss.target_regression import REGLoss

class FcotOnlineRegressionActor(BaseActor):
    """Actor for training the FCOT network."""
    def __init__(self, net, objective, device, loss_weight=None):
        super().__init__(net, objective, device)
        if loss_weight is None:
            loss_weight = {'test_reg_72': 1, 'test_init_reg_72': 0, 'test_iter_reg_72': 1}
        self.loss_weight = loss_weight
        self.dense_iou_pred = DenseIouPred(dim=4)
        self.iou_loss = REGLoss(dim=4, loss_type='iou')

    def __call__(self, data, gen_iou_map=False):
        """
        args:
            data - The input data, should contain the fields 'train_images', 'test_images', 'train_anno',
                    'test_proposals', 'proposal_iou' and 'test_label'.

        returns:
            loss    - the training loss
            stats  -  dict containing detailed losses
        """
        # Run network
        target_scores_18, target_scores_72, offset_maps = self.net(train_imgs=data['train_images'],
                                                                   test_imgs=data['test_images'],
                                                                   train_bb=data['train_anno'])

        if gen_iou_map:
            pred_iou_map = self.dense_iou_pred(offset_maps[-1], data['ind_72'], data['w2h2_72'], radius=5)
        else:
            pred_iou_map = None

        # Regression losses
        reg_losses_test_72 = [self.objective['reg_72'](w2h2, data['ind_72'], data['w2h2_72'], radius=2) for w2h2 in offset_maps]

        # Loss of the final filter
        reg_loss_test_72 = reg_losses_test_72[-1]
        loss_reg_72 = self.loss_weight['test_reg_72'] * reg_loss_test_72

        # Loss for the initial filter iteration
        loss_test_init_reg_72 = 0
        if 'test_init_reg_72' in self.loss_weight.keys():
            loss_test_init_reg_72 = self.loss_weight['test_init_reg_72'] * reg_losses_test_72[0]

        # Loss for the intermediate filter iterations
        loss_test_iter_reg_72 = 0
        if 'test_iter_reg_72' in self.loss_weight.keys():
            test_iter_weights_72 = self.loss_weight['test_iter_reg_72']
            if isinstance(test_iter_weights_72, list):
                loss_test_iter_reg_72 = sum([a * b for a, b in zip(test_iter_weights_72, reg_losses_test_72[1:-1])])
            else:
                loss_test_iter_reg_72 = (test_iter_weights_72 / (len(reg_losses_test_72) - 2)) * sum(
                    reg_losses_test_72[1:-1])

        # Total loss
        loss = loss_reg_72 + loss_test_iter_reg_72

        # Log stats
        stats = {'Loss/total': loss.item(),
                 'Loss/reg_72': loss_reg_72.item(),
                }
        if 'test_init_reg_72' in self.loss_weight.keys():
            stats['Loss/test_init_reg_72'] = loss_test_init_reg_72.item()
        if 'test_iter_reg_72' in self.loss_weight.keys():
            stats['Loss/test_iter_reg_72'] = loss_test_iter_reg_72.item()
        stats['RegTrain/test_loss_72'] = reg_loss_test_72.item()
        if len(reg_losses_test_72) > 0:
            stats['RegTrain/test_init_loss'] = reg_losses_test_72[0].item()
            if len(reg_losses_test_72) > 2:
                stats['RegTrain/test_iter_loss'] = sum(reg_losses_test_72[1:-1]).item() / (len(reg_losses_test_72) - 2)

        return loss, stats, target_scores_72[-1][:,0, ...], data['test_images'][:,0, ...], pred_iou_map

