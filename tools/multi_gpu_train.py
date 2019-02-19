# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import tensorflow.contrib.slim as slim
import os, sys
import numpy as np
import time
sys.path.append("../")

from libs.configs import cfgs
# from libs.networks import build_whole_network2
from libs.networks import build_whole_network
from data.io.read_tfrecord_multi_gpu import next_batch
from libs.box_utils import show_box_in_tensor
from help_utils import tools

os.environ["CUDA_VISIBLE_DEVICES"] = cfgs.GPU_GROUP


def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
       List of pairs of (gradient, variable) where the gradient has been averaged
       across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def sum_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
       List of pairs of (gradient, variable) where the gradient has been averaged
       across all towers.
    """
    sum_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_sum(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        sum_grads.append(grad_and_var)
    return sum_grads


def get_gtboxes_and_label(gtboxes_and_label, num_objects):
    return gtboxes_and_label[:int(num_objects), :]


def train():

    with tf.Graph().as_default(), tf.device('/cpu:0'):

        global_step = slim.get_or_create_global_step()
        lr = tf.train.piecewise_constant(global_step,
                                         boundaries=[np.int64(cfgs.DECAY_STEP[0]), np.int64(cfgs.DECAY_STEP[1]),
                                                     np.int64(cfgs.DECAY_STEP[2])],
                                         values=[cfgs.LR, cfgs.LR / 10., cfgs.LR / 100., cfgs.LR / 1000.])

        optimizer = tf.train.MomentumOptimizer(lr, momentum=cfgs.MOMENTUM)
        faster_rcnn = build_whole_network.DetectionNetwork(base_network_name=cfgs.NET_NAME,
                                                           is_training=True)

        with tf.name_scope('get_batch'):
            num_gpu = len(cfgs.GPU_GROUP.strip().split(','))
            img_name_batch, img_batch, gtboxes_and_label_batch, num_objects_batch, img_h_batch, img_w_batch = \
                next_batch(dataset_name=cfgs.DATASET_NAME,  # 'pascal', 'coco'
                           batch_size=cfgs.BATCH_SIZE * num_gpu,
                           shortside_len=cfgs.IMG_SHORT_SIDE_LEN,
                           is_training=True)
            # gtboxes_and_label = tf.reshape(gtboxes_and_label_batch, [-1, 5])
            # if cfgs.NET_NAME in ['resnet101_v1d', 'resnet50_v1d']:
            #     img_batch = img_batch / tf.constant([cfgs.PIXEL_STD])

        # data processing
        inputs_list = []
        for i in range(num_gpu):
            # img_name = img_name_batch[i]
            img = tf.expand_dims(img_batch[i], axis=0)

            gtboxes_and_label = tf.cast(tf.reshape(gtboxes_and_label_batch[i], [-1, 5]), tf.float32)
            num_objects = num_objects_batch[i]
            num_objects = tf.cast(tf.reshape(num_objects, [-1, ]), tf.float32)

            img_h = img_h_batch[i]
            img_w = img_w_batch[i]
            # img_h = tf.cast(tf.reshape(img_h, [-1, ]), tf.float32)
            # img_w = tf.cast(tf.reshape(img_w, [-1, ]), tf.float32)

            inputs_list.append([img, gtboxes_and_label, num_objects, img_h, img_w])

        # put_op_list = []
        # get_op_list = []
        # for i in range(num_gpu):
        #     with tf.device("/GPU:%s" % i):
        #         area = tf.contrib.staging.StagingArea(
        #             dtypes=[tf.float32, tf.float32, tf.float32])
        #         put_op_list.append(area.put(inputs_list[i]))
        #         get_op_list.append(area.get())

        tower_grads = []
        biases_regularizer = tf.no_regularizer
        weights_regularizer = tf.contrib.layers.l2_regularizer(cfgs.WEIGHT_DECAY)

        total_loss_dict = {
            'rpn_cls_loss': tf.constant(0., tf.float32),
            'rpn_loc_loss': tf.constant(0., tf.float32),
            'fastrcnn_cls_loss': tf.constant(0., tf.float32),
            'fastrcnn_loc_loss': tf.constant(0., tf.float32),
            'total_losses': tf.constant(0., tf.float32),

        }

        with tf.variable_scope(tf.get_variable_scope()):
            for i in range(num_gpu):
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('tower_%d' % i):
                        with slim.arg_scope(
                                [slim.model_variable, slim.variable],
                                device='/device:CPU:0'):
                            with slim.arg_scope([slim.conv2d, slim.conv2d_in_plane,
                                                 slim.conv2d_transpose, slim.separable_conv2d, slim.fully_connected],
                                                weights_regularizer=weights_regularizer,
                                                biases_regularizer=biases_regularizer,
                                                biases_initializer=tf.constant_initializer(0.0)):

                                gtboxes_and_label = tf.py_func(get_gtboxes_and_label,
                                                               inp=[inputs_list[i][1], inputs_list[i][2]],
                                                               Tout=tf.float32)
                                gtboxes_and_label = tf.reshape(gtboxes_and_label, [-1, 5])

                                img = inputs_list[i][0]
                                img_shape = inputs_list[i][-2:]
                                img = tf.image.crop_to_bounding_box(image=img,
                                                                    offset_height=0,
                                                                    offset_width=0,
                                                                    target_height=tf.cast(img_shape[0], tf.int32),
                                                                    target_width=tf.cast(img_shape[1], tf.int32))

                                outputs = faster_rcnn.build_whole_detection_network(input_img_batch=img,
                                                                                    gtboxes_batch=gtboxes_and_label)
                                gtboxes_in_img = show_box_in_tensor.draw_boxes_with_categories(img_batch=img,
                                                                                               boxes=gtboxes_and_label[
                                                                                                     :, :-1],
                                                                                               labels=gtboxes_and_label[
                                                                                                      :, -1])
                                tf.summary.image('Compare/gtboxes_gpu:%d' % i, gtboxes_in_img)

                                if cfgs.ADD_BOX_IN_TENSORBOARD:
                                    detections_in_img = show_box_in_tensor.draw_boxes_with_categories_and_scores(
                                        img_batch=img,
                                        boxes=outputs[0],
                                        scores=outputs[1],
                                        labels=outputs[2])
                                    tf.summary.image('Compare/final_detection_gpu:%d' % i, detections_in_img)

                                loss_dict = outputs[-1]

                                total_losses = 0.0
                                for k in loss_dict.keys():
                                    total_losses += loss_dict[k]
                                    total_loss_dict[k] += loss_dict[k] / num_gpu

                                total_losses = total_losses / num_gpu
                                total_loss_dict['total_losses'] += total_losses

                                if i == num_gpu - 1:
                                    regularization_losses = tf.get_collection(
                                        tf.GraphKeys.REGULARIZATION_LOSSES)
                                    # weight_decay_loss = tf.add_n(slim.losses.get_regularization_losses())
                                    total_losses = total_losses + tf.add_n(regularization_losses)

                        tf.get_variable_scope().reuse_variables()
                        grads = optimizer.compute_gradients(total_losses)
                        tower_grads.append(grads)

        for k in total_loss_dict.keys():
            tf.summary.scalar('{}/{}'.format(k.split('_')[0], k), total_loss_dict[k])

        if len(tower_grads) > 1:
            grads = sum_gradients(tower_grads)
        else:
            grads = tower_grads[0]

        # final_gvs = []
        # with tf.variable_scope('Gradient_Mult'):
        #     for grad, var in grads:
        #         scale = 1.
        #         # if '/biases:' in var.name:
        #         #    scale *= 2.
        #         if 'conv_new' in var.name:
        #             scale *= 3.
        #         if not np.allclose(scale, 1.0):
        #             grad = tf.multiply(grad, scale)
        #         final_gvs.append((grad, var))

        apply_gradient_op = optimizer.apply_gradients(grads, global_step=global_step)

        variable_averages = tf.train.ExponentialMovingAverage(0.9999, global_step)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())

        train_op = tf.group(apply_gradient_op, variables_averages_op)
        # train_op = optimizer.apply_gradients(final_gvs, global_step=global_step)
        summary_op = tf.summary.merge_all()

        restorer, restore_ckpt = faster_rcnn.get_restorer()
        saver = tf.train.Saver(max_to_keep=10)

        init_op = tf.group(
            tf.global_variables_initializer(),
            tf.local_variables_initializer()
        )

        tfconfig = tf.ConfigProto(
            allow_soft_placement=True, log_device_placement=False)
        tfconfig.gpu_options.allow_growth = True
        with tf.Session(config=tfconfig) as sess:
            sess.run(init_op)

            # sess.run(tf.initialize_all_variables())
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord, sess=sess)

            summary_path = os.path.join(cfgs.SUMMARY_PATH, cfgs.VERSION)
            tools.mkdir(summary_path)
            summary_writer = tf.summary.FileWriter(summary_path, graph=sess.graph)

            if not restorer is None:
                restorer.restore(sess, restore_ckpt)
                print('restore model')

            for step in range(cfgs.MAX_ITERATION // num_gpu):
                training_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))

                # start = time.time()
                #
                # _, global_stepnp, total_loss_dict_ = \
                #     sess.run([train_op, global_step, total_loss_dict])
                #
                # end = time.time()
                #
                # print('**' * 20)
                # print("""%s: global_step%d  current_step%d"""
                #       % (training_time, global_stepnp * num_gpu, step * num_gpu))
                # print("""per_cost_time:%.3fs"""
                #       % ((end - start) / num_gpu))
                # loss_str = ''
                # for k in total_loss_dict_.keys():
                #     loss_str += '%s:%.3f\n' % (k, total_loss_dict_[k])
                # print(loss_str)

                if step % cfgs.SHOW_TRAIN_INFO_INTE != 0 and step % cfgs.SMRY_ITER != 0:
                    _, global_stepnp = sess.run([train_op, global_step])

                else:
                    if step % cfgs.SHOW_TRAIN_INFO_INTE == 0 and step % cfgs.SMRY_ITER != 0:
                        start = time.time()

                        _, global_stepnp, total_loss_dict_ = \
                            sess.run([train_op, global_step, total_loss_dict])

                        end = time.time()

                        print('***'*20)
                        print("""%s: global_step:%d  current_step:%d"""
                              % (training_time, (global_stepnp-1)*num_gpu, step*num_gpu))
                        print("""per_cost_time:%.3fs"""
                              % ((end - start) / num_gpu))
                        loss_str = ''
                        for k in total_loss_dict_.keys():
                            loss_str += '%s:%.3f\n' % (k, total_loss_dict_[k])
                        print(loss_str)

                    else:
                        if step % cfgs.SMRY_ITER == 0:
                            _, global_stepnp, summary_str = sess.run([train_op, global_step, summary_op])
                            summary_writer.add_summary(summary_str, (global_stepnp-1)*num_gpu)
                            summary_writer.flush()

                if (step > 0 and step % (cfgs.SAVE_WEIGHTS_INTE // num_gpu) == 0) or (step >= cfgs.MAX_ITERATION // num_gpu - 1):

                    save_dir = os.path.join(cfgs.TRAINED_CKPT, cfgs.VERSION)
                    if not os.path.exists(save_dir):
                        os.mkdir(save_dir)

                    save_ckpt = os.path.join(save_dir, 'coco_' + str((global_stepnp-1)*num_gpu) + 'model.ckpt')
                    saver.save(sess, save_ckpt)
                    print(' weights had been saved')

            coord.request_stop()
            coord.join(threads)


if __name__ == '__main__':

    train()










