import tensorflow as tf

from nets.nets_factory import get_network_fn


def model_fn(features, labels, mode, params):
    """
    Model function

    Args:
        features:
        labels:
        mode:
        params: [model_name, num_class, learning_rate]
    """
    print('#####################', params, params.num_class)
    network_fn = get_network_fn(params.model_name, params.num_class,
                                is_training=mode == tf.estimator.ModeKeys.TRAIN)

    logits, end_points = network_fn(features)
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

    tf.summary.scalar('learning_rate', learning_rate)
    tf.summary.image('Training data: {}'.format(labels[0]), tf.expand_dims(features[0], axis=0))

    accuracy, a_update_op = tf.metrics.accuracy(labels,
                                                tf.argmax(logits, 1),
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

    if mode == tf.estimator.ModeKeys.PREDICT:
        spec = tf.estimator.EstimatorSpec(mode=mode,
                                          predictions=end_points['predictions'])
    elif mode == tf.estimator.ModeKeys.EVAL:
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
            summary_hook = tf.train.SummarySaverHook(save_steps=1,
                                                     output_dir='./train_dir',
                                                     summary_op=tf.summary.merge_all())
            spec = tf.estimator.EstimatorSpec(mode=mode,
                                              loss=loss,
                                              train_op=train_op,
                                              training_hooks=[summary_hook])

    return spec


def create_estimator_fn(run_config, model_params):

    return tf.estimator.Estimator(model_fn=model_fn,
                                  params=model_params,
                                  config=run_config)
