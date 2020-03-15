# TF1.x Estimator
Tensorflow Estimator Training Based on TF1.13

## Requirements
Tensorflow 1.14

## Usage example

```sh
python3 run_training.py
        --model_name mobilenet_v2 \
        --image_size 224 \
        --tfrecord_path ../cifar100/train \
        --eval_tfrecord_path ../cifar100/test \
        --train_dir ./train_dir \
        --batch_size 16 \
        --train_step 10000 \
        --save_checkpoints_step 1000 \
        --num_class 100
```

## Development options

There are several training options to setup:

* enable_xla  
Enable Accelerated Linear Algebra(XLA) compiler for speeding up training.
	
* enable_mixed_precision  
Automatic porting the model to use the half-precision data type for speeding up training.

## Future works

* [ ] Metric Learning
