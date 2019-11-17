#!/bin/bash
python3 run_training.py \
		--model_name mobilenet_v2 \
		--image_size 224 \
		--tfrecord_path ../cifar100/train \
		--eval_tfrecord_path ../cifar100/test \
		--train_dir ./train_dir \
		--batch_size 16 \
		--train_step 10000 \
		--save_checkpoints_step 1000 \
		--num_class 100
