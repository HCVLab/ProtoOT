#!/bin/bash

python main.py \
  -a resnet50 \
  --batch-size 64 \
  --mlp --aug-plus \
  --data-A './datasets/OfficeHome/Art' \
  --data-B './datasets/OfficeHome/Real' \
  --num-cluster 65 \
  --epsilon 0.05\
  --sinkhorn-iterations 3\
  --temperature 0.2 \
  --exp-dir 'office-home_art-real_final' \
  --lr 0.00025 \
  --clean-model 'moco_v2_800ep_pretrain.pth.tar' \
  --epochs 200 \
  --prec-nums '1,5,15' \
  
  

