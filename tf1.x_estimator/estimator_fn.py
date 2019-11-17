import tensorflow as tf

from nets.nets_factory import get_network_fn


def model_fn(features, labels, mode, params):
    """
    Model function

    Args:
        features:
        labels:
        mode:
        params: [name, num_class, learning_rate]
    """
    print('#####################', params, params.num_class)
    network_fn = get_network_fn(params.model_name, params.num_class,
                                is_training=mode == tf.estimator.ModeKeys.TRAIN)

    logits, end_points = network_fn(features)
    one_hot = tf.one_hot(labels, depth=params.num_class)

    loss = tf.losses.softmax_cross_entropy(one_hot, logits)

    optimizer = tf.train.MomentumOptimizer(params.learning_rate,
                                           params.momentum)

    global_step = tf.train.get_or_create_global_step()
    train_op = optimizer.minimize(loss=loss,
                                  global_step=global_step)
    accuracy, a_update_op = tf.metrics.accuracy(one_hot,
                                                tf.nn.softmax(logits),
                                                name='accuracy')
    presicion, p_update_op = tf.metrics.precision_at_k(labels,
                                                       logits,
                                                       k=1,
                                                       name='presicion')
    recall, r_update_op = tf.metrics.recall_at_k(labels,
                                                 logits,
                                                 k=1,
                                                 name='recall')

    eval_metric_ops = {'accuracy': (accuracy, a_update_op),
                       'presicion': (presicion, p_update_op),
                       'recall': (recall, r_update_op)}

    eval_logging_hook = tf.train.LoggingTensorHook({'accuracy': accuracy,
                                                    'precision': presicion,
                                                    'recall': recall}, every_n_iter=100)

    if mode == tf.estimator.ModeKeys.PREDICT:
        spec = tf.estimator.EstimatorSpec(mode,
                                          predictions=end_points['predictions'])
    elif mode == tf.estimator.ModeKeys.EVAL:
        spec = tf.estimator.EstimatorSpec(tf.estimator.ModeKeys.TRAIN,
                                          loss=loss,
                                          train_op=train_op,
                                          eval_metric_ops=eval_metric_ops,
                                          evaluation_hooks=[eval_logging_hook])
    elif mode == tf.estimator.ModeKeys.TRAIN:
        spec = tf.estimator.EstimatorSpec(tf.estimator.ModeKeys.TRAIN,
                                          loss=loss,
                                          train_op=train_op)
    return spec


def create_estimator_fn(run_config, model_params):

    return tf.estimator.Estimator(model_fn=model_fn,
                                  params=model_params,
                                  config=run_config)
