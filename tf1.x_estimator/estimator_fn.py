import tensorflow as tf

from nets.nets_factory import get_network_fn
from utils.eval_metrics_utils import eval_metrics
from utils.train_utils import cosine_decay_warmup
# from utils.optimizer_utils import get_optimizer
# from utils.learning_rate_utils import get_lr_strategy


def model_fn(features, labels, mode, params):
    """
    Model function

    Args:
        features:
        labels:
        mode:
        params: [model_name, num_class, learning_rate]
    """
    network_fn = get_network_fn(params.model_name, params.num_class,
                                is_training=mode == tf.estimator.ModeKeys.TRAIN)

    logits, end_points = network_fn(features)
    predictions = end_points['predictions']
    one_hot_pred = tf.one_hot(tf.arg_max(predictions, dimension=1), tf.shape(predictions)[1])
    one_hot_label = tf.one_hot(labels, depth=params.num_class)

    loss = tf.losses.softmax_cross_entropy(one_hot_label, logits)

    global_step = tf.train.get_or_create_global_step()
    learning_rate = cosine_decay_warmup(0.1, global_step, 0.01, 500, params.decay_steps)

    optimizer = tf.train.MomentumOptimizer(learning_rate,
                                           params.momentum)

    accuracy, accuracy_update_op = tf.metrics.accuracy(labels,
                                                       tf.argmax(logits, 1),
                                                       name='accuracy')
    metric_ops = eval_metrics(one_hot_pred, one_hot_label)

    if mode == tf.estimator.ModeKeys.PREDICT:
        spec = tf.estimator.EstimatorSpec(mode=mode,
                                          predictions=predictions)
    elif mode == tf.estimator.ModeKeys.EVAL:
        eval_metric_ops = {'eval/precision': metric_ops['overall/precision'],
                           'eval/recall': metric_ops['overall/recall'],
                           'accuracy': (accuracy, accuracy_update_op)}

        spec = tf.estimator.EstimatorSpec(mode=mode,
                                          loss=loss,
                                          eval_metric_ops=eval_metric_ops)

    elif mode == tf.estimator.ModeKeys.TRAIN:
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            grads = optimizer.compute_gradients(loss=loss)
            train_op = optimizer.apply_gradients(grads, global_step=global_step)

            if params.summary_variables_and_grads:
                for var in tf.trainable_variables():
                    tf.summary.histogram(var.op.name, var)
                for grad, var in grads:
                    if grad is not None:
                        tf.summary.histogram(var.op.name + '/gradients', grad)

            tf.summary.scalar('accuracy', accuracy_update_op)
            tf.summary.scalar('train/precision', metric_ops['overall/precision'][1])
            tf.summary.scalar('train/recall', metric_ops['overall/recall'][1])
            tf.summary.scalar('learning_rate', learning_rate)
            tf.summary.scalar('loss', loss)
            tf.summary.image('Training data', tf.expand_dims(features[0], axis=0))

            summary_hook = tf.train.SummarySaverHook(save_steps=params.summary_save_step,
                                                     output_dir=params.output_dir,
                                                     scaffold=tf.train.Scaffold(summary_op=tf.summary.merge_all()))

            spec = tf.estimator.EstimatorSpec(mode=mode,
                                              loss=loss,
                                              train_op=train_op,
                                              training_hooks=[summary_hook])

    return spec


def create_estimator_fn(run_config, model_params):

    return tf.estimator.Estimator(model_fn=model_fn,
                                  params=model_params,
                                  config=run_config)
