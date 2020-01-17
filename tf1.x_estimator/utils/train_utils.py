import numpy as np

import tensorflow as tf


def cosine_decay_warmup(start_lr, global_step, warmup_base_lr, warmup_step, decay_step, alpha=0.0):
    decay_step = tf.cast(decay_step, dtype=tf.int64)
    cosine_decay = 0.5 * (1. + tf.math.cos(np.pi * tf.cast((global_step - warmup_step) / (decay_step - warmup_step), dtype=tf.float32)))
    decayed = (1. - alpha) * cosine_decay + alpha
    cosine_lr = start_lr * decayed
    
    current_step_prop = tf.cast(global_step / warmup_step, dtype=tf.float32)
    warmup_lr = tf.cast(current_step_prop * (start_lr - warmup_base_lr) + warmup_base_lr, dtype=tf.float32)

    return tf.where(global_step > warmup_step,
                    cosine_lr,
                    warmup_lr)


def exponential_decay_warmup(start_lr, global_step, warmup_base_lr, warmup_step, decay_step, decay_rate=0.94):
    decay_step = tf.cast(decay_step, dtype=tf.int64)
    exp_lr = start_lr * decay_rate ** tf.cast((global_step - warmup_step) / (decay_step - warmup_step), dtype=tf.float32)

    current_step_prop = tf.cast(global_step / warmup_step, dtype=tf.float32)
    warmup_lr = tf.cast(current_step_prop * (start_lr - warmup_base_lr) + warmup_base_lr, dtype=tf.float32)

    return tf.where(global_step > warmup_step,
                    exp_lr,
                    warmup_lr)


def constant_lr_warmup(constant_lr, global_step, warmup_base_lr, warmup_step):
    warmup_lr = tf.cast(global_step / warmup_step * (constant_lr - warmup_base_lr) + warmup_base_lr, dtype=tf.float32)

    return tf.where(global_step > warmup_step,
                    constant_lr,
                    warmup_lr)
