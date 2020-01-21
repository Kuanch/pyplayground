import tensorflow as tf

from dataset.dataset_factory import get_dataset_fn


def create_input_fn(tfrecord_path,
                    batch_size,
                    preprocess_name,
                    get_preprocess_fn,
                    image_size=224,
                    is_training=True,
                    enable_rand_augment=False):
    preprocess_fn = get_preprocess_fn(preprocess_name)
    dataset_fn = get_dataset_fn(tfrecord_path,
                                batch_size,
                                preprocess_fn,
                                image_size,
                                is_training=is_training,
                                enable_rand_augment=enable_rand_augment)
    return dataset_fn


def create_train_and_eval_specs(train_in_fn,
                                eval_in_fn,
                                max_steps):
    train_spec = tf.estimator.TrainSpec(input_fn=train_in_fn,
                                        max_steps=max_steps)

    eval_spec = tf.estimator.EvalSpec(input_fn=eval_in_fn, steps=10, throttle_secs=60)

    return train_spec, eval_spec
