#! /bin/bash

cd ../ltr

#WORKSPACE_STAGE1="/home/cyt/data/experiments/FCOT-jointly"
#WORKSPACE_STAGE2="/home/cyt/data/experiments/FCOT-jointly-online-reg"
WORKSPACE_STAGE1="/your/workspace/for/trianing/stage1"
WORKSPACE_STAGE2="/your/workspace/for/trianing/stage2"
LOAD_MODEL_EPOCH=70

# stage1: train backbone, classifier-72, classifier-18 and regression branch (except for optimizer)
python run_training.py fcot fcot \
  --samples_per_epoch 40000 \
  --use_pretrained_dimp 'True' \
  --pretrained_dimp50 "../models/dimp50.pth" \
  --train_cls_72_and_reg_init 'True' \
  --train_cls_18 'True' \
  --workspace_dir $WORKSPACE_STAGE1 \
  --lasot_rate 1 \
  --total_epochs 70 \
  --batch_size 40 \
  --devices_id 0 1 2 3 4 5 6 7  # used gpus

# stage2: train regression optimizer
python run_training.py fcot fcot \
  --samples_per_epoch 26000 \
  --load_model 'True' \
  --fcot_model "${WORKSPACE_STAGE1}/checkpoints/ltr/fcot/fcot/FCOTNet_ep00${LOAD_MODEL_EPOCH}.pth.tar"\
  --train_reg_optimizer 'True' \
  --workspace_dir $WORKSPACE_STAGE2 \
  --total_epochs 5 \
  --batch_size 40 \
  --devices_id 0 1 2 3 4 5 6 7  # used gpus