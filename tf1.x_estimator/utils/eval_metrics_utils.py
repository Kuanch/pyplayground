import tensorflow as tf


def eval_metrics(one_hot_preds, one_hot_labels):
    one_hot_preds = tf.cast(one_hot_preds, dtype=tf.bool)
    one_hot_labels = tf.cast(one_hot_labels, dtype=tf.bool)

    # Get indics of correct predictions and sum by column
    num_correct_per_class = tf.squeeze(tf.reduce_sum(tf.cast(tf.logical_and(one_hot_preds, one_hot_labels), dtype=tf.float32), axis=0))

    num_predict_per_class = tf.reduce_sum(
        tf.cast(one_hot_preds, dtype=tf.float32), axis=0)

    num_label_per_class = tf.reduce_sum(
        tf.cast(one_hot_labels, dtype=tf.float32), axis=0)

    predict_nonzero_indice = tf.squeeze(tf.where(tf.math.greater(
        num_predict_per_class, 0)))

    label_nonzero_indice = tf.squeeze(tf.where(tf.math.greater(
        num_label_per_class, 0)))
    # Prediction per class could be zero, take only noe zero elements
    correct_pred_per_class = tf.gather(num_correct_per_class, predict_nonzero_indice)
    predict_per_class = tf.gather(num_predict_per_class, predict_nonzero_indice)

    # Label per class could be zero, take only noe zero elements
    correct_label_per_class = tf.gather(num_correct_per_class, label_nonzero_indice)
    label_per_class = tf.gather(num_label_per_class, label_nonzero_indice)

    metrics = {}
    metrics['overall/precision'] = tf.metrics.mean(correct_pred_per_class / predict_per_class)
    metrics['overall/recall'] = tf.metrics.mean(correct_label_per_class / label_per_class)

    return metrics
