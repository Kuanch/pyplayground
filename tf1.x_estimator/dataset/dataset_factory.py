import tensorflow as tf


def parse_fn(proto):
    keys_to_features = {'image/encoded': tf.io.FixedLenFeature([], tf.string),
                        'image/class/label': tf.io.FixedLenFeature([], tf.int64)}

    parsed_features = tf.io.parse_single_example(proto, keys_to_features)

    parsed_features['decoded_image'] = tf.io.decode_jpeg(parsed_features['image/encoded'], channels=3)

    parsed_features['category_id'] = tf.cast(parsed_features['image/class/label'], dtype=tf.int64)

    return parsed_features['decoded_image'], parsed_features['category_id']


def dataset_fn(tfrecord_path,
               batch_size,
               preprocess_fn,
               image_size,
               is_training):
    dataset_list = tf.data.Dataset.list_files(tfrecord_path + '/*')

    dataset = dataset_list.interleave(tf.data.TFRecordDataset,
                                      cycle_length=8,
                                      num_parallel_calls=8)

    dataset = dataset.map(parse_fn, num_parallel_calls=8)

    dataset = dataset.map(lambda image, label:
                          preprocess_fn(image,
                                        label,
                                        image_size,
                                        image_size,
                                        is_training=is_training),
                          num_parallel_calls=8)

    dataset = dataset.shuffle(buffer_size=8)

    dataset = dataset.repeat()

    dataset = dataset.batch(batch_size)

    dataset = dataset.prefetch(5)

    return dataset


def get_dataset_fn(tfrecord_path,
                   batch_size,
                   preprocess_fn,
                   image_size,
                   is_training):
    def _get_dataset_fn():
        return dataset_fn(tfrecord_path,
                          batch_size,
                          preprocess_fn,
                          image_size,
                          is_training)
    return _get_dataset_fn
