import tensorflow as tf


def eval_metrics(one_hot_preds, one_hot_labels, is_training=True):
    one_hot_preds = tf.cast(one_hot_preds, dtype=tf.bool)
    one_hot_labels = tf.cast(one_hot_labels, dtype=tf.bool)

    num_correct_per_class = tf.reduce_sum(
        tf.cast(tf.logical_and(one_hot_preds, one_hot_labels),
                tf.float32), axis=0)

    num_predict_per_class = tf.reduce_sum(
        tf.cast(one_hot_preds, dtype=tf.float32), axis=0)

    num_label_per_class = tf.reduce_sum(
        tf.cast(one_hot_labels, dtype=tf.float32), axis=0)

    nonzero_indice = tf.squeeze(tf.where(tf.math.greater(
        num_label_per_class, 0)))

    num_correct = tf.reduce_sum(num_correct_per_class)
    num_predict = tf.reduce_sum(num_predict_per_class)
    num_label = tf.reduce_sum(num_label_per_class)

    metrics = {}
    precision = num_correct / num_predict
    recall = num_correct / num_label

    class_precision = tf.reduce_mean(
        tf.gather(num_correct_per_class, nonzero_indice) /
        tf.gather(num_predict_per_class, nonzero_indice))

    class_recall = tf.reduce_mean(
        tf.gather(num_correct_per_class, nonzero_indice) /
        tf.gather(num_label_per_class, nonzero_indice))

    if is_training:
        metrics['overall/precision'] = precision
        metrics['overall/recall'] = recall
        metrics['class/precision'] = class_precision
        metrics['class/recall'] = class_recall

    else:
        metrics['overall/precision'] = tf.metrics.mean(precision, name='precision')
        metrics['overall/recall'] = tf.metrics.mean(recall, name='recall')
        metrics['class/precision'] = tf.metrics.mean(class_precision, name='class_prescision')
        metrics['class/recall'] = tf.metrics.mean(class_recall, name='class_recal')

    return metrics
