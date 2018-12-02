# -*- coding: utf-8 -*-

'''
this file is to convert pascal to tfrecord
'''

import numpy as np
import cv2
import os, sys
import tensorflow as tf
import xml.etree.cElementTree as ET
from libs.label_name_dict.label_dict import NAME_LABEL_MAP


tf.app.flags.DEFINE_string('VOC_dir', '/home/yjr/DataSet/VOC', 'Voc dir ')
FLAGS = tf.app.flags.FLAGS

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def read_xml_target_box_and_label(xml_path):
    '''

    :param xml_path:
    :return:img_height, img_width, gtboxes
    gtboxes is a array of shape [num_of_gtboxes, 5]
    a row in gtboxes is [xmin. ymin. xmax, ymax, label]
    '''
    tree = ET.parse(xml_path)
    root = tree.getroot()
    img_width = None
    img_height = None
    box_list = []
    for child_of_root in root:
        if child_of_root.tag == 'filename':
            assert child_of_root.text == xml_path.split('/')[-1].split('.')[0] + '.jpg', 'xml_name and img_name cannot match'
        if child_of_root.tag == 'size':
            for child_item in child_of_root:
                if child_item.tag == 'width':
                    img_width = int(child_item.text)
                if child_item.tag == 'height':
                    img_height = int(child_item.text)
        if child_of_root.tag == 'object':
            label = None
            for child_item in child_of_root:
                if child_item.tag == 'name':
                    # print child_item.text
                    label = NAME_LABEL_MAP[child_item.text]
                if child_item.tag == 'bndbox':
                    tmp_box = []
                    for node in child_item:
                        tmp_box.append(int(node.text))  # [xmin, ymin. xmax, ymax]
                    assert label is not None, 'label is none, error'
                    tmp_box.append(label) #[xmin, ymin, xmax, ymax, label]
                    box_list.append(tmp_box)

    gtbox_list = np.array(box_list, dtype=np.int32)  # [xmin, ymin, xmax, ymax, label]

    xmin, ymin, xmax, ymax, label = gtbox_list[:, 0], gtbox_list[:, 1], gtbox_list[:, 2], gtbox_list[:, 3],\
                                    gtbox_list[:, 4]

    gtbox_list = np.transpose(np.stack([xmin, ymin, xmax, ymax, label], axis=0))  # [xmin, ymin, xmax, ymax, label]
    # print gtbox_list.shape
    return img_height, img_width, gtbox_list

def convert_pascal(dataset_name):

    dataset_rootdir = os.path.join(FLAGS.VOC_dir, 'VOCtrain_val/VOC2007') if dataset_name == 'train' \
        else os.path.join(FLAGS.VOC_dir, 'VOC_test/VOC2007')

    imgname_list = []
    part_name = 'trainval.txt' if dataset_name == 'train' else 'test.txt'
    with open(os.path.join(dataset_rootdir, 'ImageSets/Main/aeroplane_'+part_name)) as f:
        all_lines = f.readlines()

    for a_line in all_lines:
        imgname_list.append(a_line.split()[0].strip())

    # writer_options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB)
    # writer = tf.python_io.TFRecordWriter(path='../data/tfrecords/pascal_'+dataset_name+'.tfrecord', options=writer_options)
    writer = tf.python_io.TFRecordWriter(path='../tfrecord/pascal_' + dataset_name + '.tfrecord')
    for i, img_name in enumerate(imgname_list):
        img_np = cv2.imread(os.path.join(dataset_rootdir, 'JPEGImages/'+img_name+'.jpg'))
        # if img_np == None:
        #     print img_name
        img_np = img_np[:, :, ::-1]
        assert img_np is not None, 'read img erro, imgnp is None'
        xml_path = os.path.join(dataset_rootdir, 'Annotations/'+img_name+'.xml')
        img_height, img_width, gtboxes = read_xml_target_box_and_label(xml_path)

        example = tf.train.Example(features=tf.train.Features(feature={
            'img_name': _bytes_feature(img_name),
            'img_height': _int64_feature(img_height),
            'img_width': _int64_feature(img_width),
            'img': _bytes_feature(img_np.tostring()),
            'gtboxes_and_label': _bytes_feature(gtboxes.tostring()),
            'num_objects': _int64_feature(gtboxes.shape[0])
        }))
        writer.write(example.SerializeToString())
        if i % 100 == 0:
            print('{} {} imgs convert over'.format(i, dataset_name))
    print(20*"@")
    print('all {} imgs convert over, the num is {}'.format(dataset_name, i))

if __name__ == '__main__':
    # w, h, gtboxes = read_xml_target_box_and_label('/home/yjr/DataSet/VOC/VOCtrain_val/VOC2007/Annotations/000005.xml')
    # print w, h
    # print gtboxes
    convert_pascal('train')
    convert_pascal('test')

