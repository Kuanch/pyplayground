import tensorflow as tf

from nets.nets_factory import get_network_fn
from utils.eval_metrics_utils import eval_metrics


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
    one_hot = tf.one_hot(labels, depth=params.num_class)

    loss = tf.losses.softmax_cross_entropy(one_hot, logits)

    global_step = tf.train.get_or_create_global_step()
    learning_rate = tf.train.cosine_decay_restarts(params.learning_rate,
                                                   global_step,
                                                   first_decay_steps=500,
                                                   m_mul=0.9,
                                                   alpha=0.0001)

    optimizer = tf.train.MomentumOptimizer(learning_rate,
                                           params.momentum)

    accuracy, a_update_op = tf.metrics.accuracy(labels,
                                                tf.argmax(logits, 1),
                                                name='accuracy')

    tf.summary.scalar('learning_rate', learning_rate)
    tf.summary.image('Training data', tf.expand_dims(features[0], axis=0))

    if mode == tf.estimator.ModeKeys.PREDICT:
        spec = tf.estimator.EstimatorSpec(mode=mode,
                                          predictions=predictions)
    elif mode == tf.estimator.ModeKeys.EVAL:
        metric_ops = eval_metrics(predictions, one_hot, is_training=False)
        eval_metric_ops = {'overall/precision': metric_ops['overall/precision'],
                           'overall/recall': metric_ops['overall/recall'],
                           'class/precision': metric_ops['class/precision'],
                           'class/recall': metric_ops['class/recall'],
                           'accuracy': (accuracy, a_update_op)}
        tf.summary.scalar('presicion', eval_metric_ops['overall/precision'])
        tf.summary.scalar('recall', eval_metric_ops['overall/recall'])
        tf.summary.scalar('class/presicion', eval_metric_ops['class/precision'])
        tf.summary.scalar('class/recall', eval_metric_ops['class/recall'])
        tf.summary.scalar('accuracy', eval_metric_ops['accuracy'])

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

            spec = tf.estimator.EstimatorSpec(mode=mode,
                                              loss=loss,
                                              train_op=train_op)

    return spec


def create_estimator_fn(run_config, model_params):

    return tf.estimator.Estimator(model_fn=model_fn,
                                  params=model_params,
                                  config=run_config)
