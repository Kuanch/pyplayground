import os
import argparse
from collections import namedtuple

import tensorflow as tf

from input_fn import create_input_fn
from input_fn import create_train_and_eval_specs
from estimator_fn import create_estimator_fn
from preprocess.preprocessing_factory import get_preprocessing


ModelParams = namedtuple('ModelParams', ['model_name',
                                         'num_class',
                                         'learning_rate',
                                         'momentum',
                                         'summary_variables_and_grads',
                                         'summary_save_step',
                                         'output_dir',
                                         'decay_steps',
                                         'decay_rate',
                                         'warmup_base_lr',
                                         'warmup_step',
                                         'label_smoothing'])


def main(args):
    tf.logging.set_verbosity(tf.logging.INFO)
    train_input_fn = create_input_fn(args.tfrecord_path,
                                     args.batch_size,
                                     args.model_name,
                                     image_size=args.image_size,
                                     get_preprocess_fn=get_preprocessing,
                                     enable_rand_augment=args.enable_rand_augment)
    eval_input_fn = create_input_fn(args.eval_tfrecord_path,
                                    batch_size=args.eval_batch_size,
                                    image_size=args.image_size,
                                    preprocess_name=args.model_name,
                                    get_preprocess_fn=get_preprocessing,
                                    is_training=False,
                                    enable_rand_augment=args.enable_rand_augment)

    train_spec, eval_spec = create_train_and_eval_specs(train_input_fn,
                                                        eval_input_fn,
                                                        max_steps=args.train_step)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    if args.enable_xla:
        os.environ['TF_XLA_FLAGS']='--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit'
        config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    run_config = tf.estimator.RunConfig(args.train_dir,
                                        save_checkpoints_steps=args.save_checkpoints_step,
                                        keep_checkpoint_max=args.keep_checkpoint_max,
                                        session_config=config)
    model_params = ModelParams(model_name=args.model_name,
                               num_class=args.num_class,
                               learning_rate=args.learning_rate,
                               momentum=args.momentum,
                               summary_variables_and_grads=args.summary_variables_and_grads,
                               summary_save_step=args.summary_save_step,
                               output_dir=args.train_dir,
                               decay_steps=args.train_step,
                               decay_rate=0.1,
                               warmup_base_lr=args.warmup_base_lr,
                               warmup_step=args.warmup_step,
                               label_smoothing=args.label_smoothing)
    estimator = create_estimator_fn(run_config, model_params)

    tf.estimator.train_and_evaluate(estimator,
                                    train_spec,
                                    eval_spec)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', default=1, type=int)
    parser.add_argument('--train_step', default=1000, type=int)
    parser.add_argument('--num_images', default=1000, type=int)
    parser.add_argument('--image_size', default=224, type=int)
    parser.add_argument('--gpu_id', default='0')
    parser.add_argument('--tfrecord_path', type=str)
    parser.add_argument('--eval_tfrecord_path', type=str)
    parser.add_argument('--train_dir', default='./train_dir', type=str)
    parser.add_argument('--keep_checkpoint_max', default=5, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--eval_batch_size', default=32, type=int)
    parser.add_argument('--num_class', default=10, type=int)
    parser.add_argument('--learning_rate', default=0.1, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--warmup_base_lr', default=0.01, type=float)
    parser.add_argument('--warmup_step', default=1000, type=int)
    parser.add_argument('--label_smoothing', default=0, type=float)
    parser.add_argument('--enable_rand_augment', dest='enable_rand_augment', action='store_true')
    parser.add_argument('--model_name', default='mobilenet_v2', type=str)
    parser.add_argument('--save_checkpoints_step', default=100, type=int)
    parser.add_argument('--summary_save_step', default=10, type=int)
    parser.add_argument('--summary_variables_and_grads', dest='summary_variables_and_grads', action='store_true')
    parser.add_argument('--enable_xla', dest='enable_xla', action='store_true')
    parser.add_argument('--enable_mixed_precision', dest='enable_mixed_precision', action='store_true')


    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    if args.enable_mixed_precision:
        os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'
    main(args)
