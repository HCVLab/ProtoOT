#!/bin/bash

python main.py \
  -a resnet50 \
  --batch-size 64 \
  --mlp --aug-plus\
  --data-A './datasets/DomainNet/images/clipart' \
  --data-B './datasets/DomainNet/images/sketch' \
  --num-cluster 7 \
  --epsilon 0.05\
  --sinkhorn-iterations 3\
  --temperature  0.2 \
  --exp-dir 'domainnet_clipart-sketch_final' \
  --lr 0.00025 \
  --clean-model 'moco_v2_800ep_pretrain.pth.tar' \
  --epochs 200 \
  --prec-nums '50,100,200' \
  

