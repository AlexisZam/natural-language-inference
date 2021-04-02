#!/usr/bin/env bash

src/run.py \
    --pretrained_model_name_or_path bert-base-cased \
    --dataset_name scitail \
    --do_train \
    --do_predict \
    --padding \
    --max_length 128 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --save_total_limit 1

# --weight_decay 0.01
