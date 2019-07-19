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
"""Generic evaluation script that evaluates a model using a given dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf
import numpy as np

from datasets import dataset_factory
from nets import nets_factory
from preprocessing import preprocessing_factory
from PIL import Image


slim = tf.contrib.slim

tf.app.flags.DEFINE_integer(
    'batch_size', 100, 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer(
    'max_num_batches', None,
    'Max number of batches to evaluate by default use all.')

tf.app.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.app.flags.DEFINE_string(
    'checkpoint_path', '/tmp/tfmodel/',
    'The directory where the model was written to or an absolute path to a '
    'checkpoint file.')

tf.app.flags.DEFINE_string(
    'eval_dir', '/tmp/tfmodel/', 'Directory where the results are saved to.')

tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 4,
    'The number of threads used to create the batches.')

tf.app.flags.DEFINE_string(
    'dataset_name', 'imagenet', 'The name of the dataset to load.')

tf.app.flags.DEFINE_string(
    'dataset_split_name', 'test', 'The name of the train/test split.')

tf.app.flags.DEFINE_string(
    'dataset_dir', None, 'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')

tf.app.flags.DEFINE_string(
    'model_name', 'inception_v3', 'The name of the architecture to evaluate.')

tf.app.flags.DEFINE_string(
    'preprocessing_name', None, 'The name of the preprocessing to use. If left '
    'as `None`, then the model_name flag is used.')

tf.app.flags.DEFINE_float(
    'moving_average_decay', None,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')

tf.app.flags.DEFINE_integer(
    'eval_image_size', None, 'Eval image size')

tf.app.flags.DEFINE_integer(
        'num_classes', 3,'num classes')
FLAGS = tf.app.flags.FLAGS


def main(_):
 
  tf.logging.set_verbosity(tf.logging.INFO)
  with tf.Graph().as_default():
    tf_global_step = slim.get_or_create_global_step()

    # ######################
    # # Select the dataset #
    # ######################
    # dataset = dataset_factory.get_dataset(
    #     FLAGS.dataset_name, FLAGS.dataset_split_name, FLAGS.dataset_dir)

    ####################
    # Select the model #
    ####################
    network_fn = nets_factory.get_network_fn(
        FLAGS.model_name,
        num_classes=(FLAGS.num_classes - FLAGS.labels_offset),
        is_training=False)

    ##############################################################
    # Create a dataset provider that loads data from the dataset #
    ##############################################################
    # provider = slim.dataset_data_provider.DatasetDataProvider(
    #     dataset,
    #     shuffle=False,
    #     common_queue_capacity=2 * FLAGS.batch_size,
    #     common_queue_min=FLAGS.batch_size)
    # [image, label, filename] = provider.get(['image', 'label', 'filename'])
    # label -= FLAGS.labels_offset

    #####################################
    # Select the preprocessing function #
    #####################################
    preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
    image_preprocessing_fn = preprocessing_factory.get_preprocessing(
        preprocessing_name,
        is_training=False)

    eval_image_size = FLAGS.eval_image_size or network_fn.default_image_size

    image = tf.placeholder(dtype=tf.float32, shape=(eval_image_size,eval_image_size,3))

    image = image_preprocessing_fn(image, eval_image_size, eval_image_size)
    
    image = tf.placeholder(dtype=tf.float32, shape=(1,eval_image_size,eval_image_size,3))

    # images, labels, filenames = tf.train.batch(
    #     [image, label, filename],
    #     batch_size=FLAGS.batch_size,
    #     num_threads=FLAGS.num_preprocessing_threads,
    #     capacity=5 * FLAGS.batch_size)

    ####################
    # Define the model #
    ####################
    logits, end_points = network_fn(image)

    if FLAGS.moving_average_decay:
      variable_averages = tf.train.ExponentialMovingAverage(
          FLAGS.moving_average_decay, tf_global_step)
      variables_to_restore = variable_averages.variables_to_restore(
          slim.get_model_variables())
      variables_to_restore[tf_global_step.op.name] = tf_global_step
    else:
      variables_to_restore = slim.get_variables_to_restore()

    predictions = tf.argmax(logits, 1)
    # labels = tf.squeeze(labels)

    # mislabeled = tf.not_equal(predictions, labels)
    # mislabeled_filenames = tf.boolean_mask(filenames, mislabeled)


    # # Define the metrics:
    # names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
    #     'Accuracy': slim.metrics.streaming_accuracy(predictions, labels),
    #     'Recall_5': slim.metrics.streaming_recall_at_k(
    #         logits, labels, 5),
    #     'Per_class_accuracy': tf.metrics.mean_per_class_accuracy(predictions, labels,
    #                                                              dataset.num_classes),
    # })

    # # Print the summaries to screen.
    # for name, value in names_to_values.items():
    #   summary_name = 'eval/%s' % name
    #   op = tf.summary.scalar(summary_name, value, collections=[])
    #   op = tf.Print(op, [value], summary_name)
    #   tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)

    #
    # # TODO(sguada) use num_epochs=1
    # if FLAGS.max_num_batches:
    #   num_batches = FLAGS.max_num_batches
    # else:
    #   # This ensures that we make a single pass over all of the data.
    #   num_batches = math.ceil(dataset.num_samples / float(FLAGS.batch_size))

    if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
      checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
    else:
      checkpoint_path = FLAGS.checkpoint_path

    tf.logging.info('Evaluating %s' % checkpoint_path)


    with tf.Session() as sess:

        saver = tf.train.Saver()
        saver.restore(sess, checkpoint_path)

        sample_images = ["/home/soumyadeep_morphle_in/tmp/data/wbc_morphle/wbc_images/neutrophils/x0y11_0.jpg"]

        from os import listdir
        from os.path import isfile, join
        import os
        in_dir = "/home/soumyadeep_morphle_in/tmp/data/wbc_morphle/wbc_images/monocytes"
        sample_images = [os.path.join(in_dir,f) for f in listdir(in_dir) if isfile(join(in_dir, f))]
        
        #with tf.Session() as sess:
        for img in sample_images:
            im = Image.open(img).resize((eval_image_size,eval_image_size))
            im = np.array(im)
            im = im.reshape(1,eval_image_size,eval_image_size,3)
      
            end_points_values, logit_values, prediction_values = sess.run([end_points, logits, predictions], feed_dict={image: im})
#      print (np.max(predict_values), np.max(logit_values))
#      print (np.argmax(predict_values), np.argmax(logit_values))

      #print(logits)
            print(end_points_values)
            exit()
            ####################	
            # Select the model #
            ###################
#            network_fn = nets_factory.get_network_fn(
#                FLAGS.model_name,
#                num_classes=(FLAGS.num_classes - FLAGS.labels_offset),
#                is_training=False)
#
#
#            #####################################
#            # Select the preprocessing function #
#            #####################################
#            preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
#            image_preprocessing_fn = preprocessing_factory.get_preprocessing(
#                preprocessing_name,
#                is_training=False)
#
#            eval_image_size = FLAGS.eval_image_size or network_fn.default_image_size
#
#	    image = tf.placeholder(dtype=tf.float32, shape=(eval_image_size,eval_image_size,3))
#
#            image = image_preprocessing_fn(image, eval_image_size, eval_image_size)
#    
#            image = tf.placeholder(dtype=tf.float32, shape=(1,eval_image_size,eval_image_size,3))
#
#            ####################
#            # Define the model #
#            ####################
#            logits, end_points = network_fn(image)
#    # eval_op = list(names_to_updates.values())
    #
    # slim.evaluation.evaluate_once(
    #     master=FLAGS.master,
    #     checkpoint_path=checkpoint_path,
    #     logdir=FLAGS.eval_dir,
    #     num_evals=num_batches,
    #     eval_op=list(names_to_updates.values()),
    #     variables_to_restore=variables_to_restore)


if __name__ == '__main__':
  tf.app.run()
  #main(1)


# checkpoint_file = 'inception_resnet_v2_2016_08_30.ckpt'
# sample_images = ['dog.jpg', 'panda.jpg']
# #Load the model
# sess = tf.Session()
# arg_scope = inception_resnet_v2_arg_scope()
# with slim.arg_scope(arg_scope):
#   logits, end_points = inception_resnet_v2(input_tensor, is_training=False)
# saver = tf.train.Saver()
# saver.restore(sess, checkpoint_file)
# for image in sample_images:
#   im = Image.open(image).resize((299,299))
#   im = np.array(im)
#   im = im.reshape(-1,299,299,3)
#   predict_values, logit_values = sess.run([end_points['Predictions'], logits], feed_dict={input_tensor: im})
#   print (np.max(predict_values), np.max(logit_values))
#   print (np.argmax(predict_values), np.argmax(logit_values))
