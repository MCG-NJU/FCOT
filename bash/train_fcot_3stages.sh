#! /bin/bash

cd ../ltr

#WORKSPACE_STAGE1="/home/cyt/data/experiments/FCOT-3stages-cls_72_and_reg_init"
#WORKSPACE_STAGE2="/home/cyt/data/experiments/FCOT-3stages-reg_optimizer"
#WORKSPACE_STAGE3="/home/cyt/data/experiments/FCOT-3stages-cls_18"
WORKSPACE_STAGE1="/your/workspace/for/trianing/stage1"
WORKSPACE_STAGE2="/your/workspace/for/trianing/stage2"
WORKSPACE_STAGE3="/your/workspace/for/trianing/stage3"
LOAD_MODEL_EPOCH1=70
LOAD_MODEL_EPOCH2=5

# stage-1: train backbone, classifier-72 and regression branch (except for optimizer)
python run_training.py fcot fcot \
  --samples_per_epoch 26000 \
  --use_pretrained_dimp 'True' \
  --pretrained_dimp50 "../models/dimp50.pth" \
  --train_cls_72_and_reg_init 'True' \
  --workspace_dir $WORKSPACE_STAGE1 \
  --norm_scale_coef 1.5 \
  --total_epochs 70 \
  --batch_size 40 \
  --devices_id 0 1 2 3 4 5 6 7  # used gpus


# stage-2: train regression optimizer
python run_training.py fcot fcot \
  --samples_per_epoch 26000 \
  --load_model 'True' \
  --fcot_model "${WORKSPACE_STAGE1}/checkpoints/ltr/fcot/fcot/FCOTNet_ep00${LOAD_MODEL_EPOCH1}.pth.tar"\
  --train_reg_optimizer 'True' \
  --workspace_dir $WORKSPACE_STAGE2 \
  --norm_scale_coef 1.5 \
  --total_epochs 5 \
  --batch_size 40 \
  --devices_id 0 1 2 3 4 5 6 7

# stage-3: train classifier-18
python run_training.py fcot fcot \
  --samples_per_epoch 26000 \
  --load_model 'True' \
  --fcot_model "${WORKSPACE_STAGE2}/checkpoints/ltr/fcot/fcot/FCOTNet_ep000${LOAD_MODEL_EPOCH2}.pth.tar"\
  --train_cls_18 'True' \
  --workspace_dir $WORKSPACE_STAGE3 \
  --norm_scale_coef 1.5 \
  --total_epochs 25 \
  --batch_size 40 \
  --devices_id 0 1 2 3 4 5 6 7