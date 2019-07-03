import os
import csv
from os.path import join
from shutil import copyfile
import tensorflow as tf
import glob
import xml.etree.ElementTree as ET

flags = tf.app.flags
flags.DEFINE_string('input_path', '', 'Root directory containing all datasets to be joined.')
flags.DEFINE_string('output_path', '', 'Root directory to store the new dataset')
FLAGS = flags.FLAGS


def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]


def copy(img_src, img_dst, label_src, label_dst):

    # copy files
    copyfile(img_src, img_dst)
    copyfile(label_src, label_dst)

    # update path in label[.xml] file

    tree = ET.parse(label_dst)
    root = tree.getroot()
    elem = root.find('path')
    elem.text = img_dst
    tree.write(label_dst)

    img_name = img_dst.split('/')[-1]
    elem = root.find('filename')
    elem.text = img_name
    tree.write(label_dst)


def join_datasets(input_path, output_path):
    uid = 1

    dirs = get_immediate_subdirectories(input_path)

    # create new structure in output path
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    out_img_path = join(output_path, 'IMG')

    if not os.path.exists(out_img_path):
        os.mkdir(out_img_path)

    out_voc_path = join(output_path, 'voc-labels')

    if not os.path.exists(out_voc_path):
        os.mkdir(out_voc_path)

    # for each dataset inside the root directory
    for dataset in dirs:
        path = join(input_path, dataset)
        in_img_path = join(path, 'IMG')
        in_voc_path = join(path, 'voc-labels')

        annotations = glob.glob(in_voc_path + '/*.xml')

        for annotation in annotations:
            tree = ET.parse(annotation)
            root = tree.getroot()

            in_img_path = root.find('path').text

            if not os.path.isfile(in_img_path):
                raise Exception('{} does not exist'.format(in_img_path))

            out_img_path_full = join(out_img_path, 'image{uid}.jpg'.format(uid=uid))
            out_voc_path_full = join(out_voc_path, 'image{uid}.xml'.format(uid=uid))

            copy(in_img_path, out_img_path_full, annotation, out_voc_path_full)

            uid += 1


def main(_):
    join_datasets(input_path=FLAGS.input_path,
                  output_path=FLAGS.output_path)


if __name__ == '__main__':
    tf.app.run()
