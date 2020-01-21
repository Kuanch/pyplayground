# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains a factory for building various models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from preprocess import inception_preprocessing
from preprocess import vgg_preprocessing
from preprocess import eff_preprocessing
from preprocess import auto_augment


def get_preprocessing(name):
    """Returns preprocessing_fn(image, height, width, **kwargs).

    Args:
      name: The name of the preprocessing function.
      is_training: `True` if the model is being used for training and `False`
        otherwise.
      use_grayscale: Whether to convert the image from RGB to grayscale.

    Returns:
      preprocessing_fn: A function that preprocessing a single image (pre-batch).
        It has the following signature:
          image = preprocessing_fn(image, output_height, output_width, ...).

    Raises:
      ValueError: If Preprocessing `name` is not recognized.
    """
    preprocessing_fn_map = {
        'inception': inception_preprocessing,
        'inception_v1': inception_preprocessing,
        'inception_v2': inception_preprocessing,
        'inception_v3': inception_preprocessing,
        'inception_v4': inception_preprocessing,
        'inception_resnet_v2': inception_preprocessing,
        'mobilenet_v1': inception_preprocessing,
        'mobilenet_v2': inception_preprocessing,
        'mobilenet_v2_035': inception_preprocessing,
        'mobilenet_v2_140': inception_preprocessing,
        'nasnet_mobile': inception_preprocessing,
        'nasnet_large': inception_preprocessing,
        'pnasnet_mobile': inception_preprocessing,
        'pnasnet_large': inception_preprocessing,
        'resnet_v1_50': vgg_preprocessing,
        'resnet_v1_101': vgg_preprocessing,
        'resnet_v1_152': vgg_preprocessing,
        'resnet_v1_200': vgg_preprocessing,
        'resnet_v2_50': vgg_preprocessing,
        'resnet_v2_101': vgg_preprocessing,
        'resnet_v2_152': vgg_preprocessing,
        'resnet_v2_200': vgg_preprocessing,
        'vgg': vgg_preprocessing,
        'vgg_a': vgg_preprocessing,
        'vgg_16': vgg_preprocessing,
        'vgg_19': vgg_preprocessing,
        'efficientnet-b0': eff_preprocessing,
        'efficientnet-b1': eff_preprocessing,
        'efficientnet-b2': eff_preprocessing,
        'efficientnet-b3': eff_preprocessing,
        'efficientnet-b4': eff_preprocessing,
        'efficientnet-b5': eff_preprocessing,
        'efficientnet-b6': eff_preprocessing,
        'efficientnet-b7': eff_preprocessing,
        'efficientnet-b8': eff_preprocessing,
        'efficientnet-l2': eff_preprocessing,
    }

    if name not in preprocessing_fn_map:
        raise ValueError('Preprocessing name [%s] was not recognized' % name)

    def preprocessing_fn(image, label, output_height, output_width, is_training, enable_rand_augment, **kwargs):
        if enable_rand_augment:
            image = tf.expand_dims(image, 0)
            image = tf.image.resize_bilinear(image, [output_height, output_width], align_corners=False)
            image = tf.squeeze(image)
            if is_training:
                image = auto_augment.distort_image_with_randaugment(image, 2, 5)
        else:
            image, label = preprocessing_fn_map[name].preprocess_image(
                image,
                label,
                output_height,
                output_width,
                is_training=is_training,
                **kwargs)
        return image, label

    return preprocessing_fn
