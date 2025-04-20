#!/bin/bash

PY_ARGS=${@:1} # Any other arguments

python main_finetune.py \
  --model AIDE \
  --blr 5e-4 \
  --epochs 5 \
  --device cuda:0 \
  --eval True \
  --eval_data_path /content/drive/MyDrive/Real_fake_dataset \
  --batch_size 16 \
  --num_workers 0 \
  --resume /content/drive/MyDrive/model/sd14_train.pth \
  --output_dir results/sd14_train \
  ${PY_ARGS}
