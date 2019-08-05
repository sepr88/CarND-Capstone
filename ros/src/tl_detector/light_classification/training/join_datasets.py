import os
from os.path import join
import tensorflow as tf
import glob
import xml.etree.ElementTree as ET
from utils.tl_utils import get_immediate_subdirectories, copy_img_label

flags = tf.app.flags
flags.DEFINE_string('input_path', '', 'Root directory containing all datasets to be joined.')
flags.DEFINE_string('output_path', '', 'Root directory to store the new dataset')
flags.DEFINE_string('img_dir', 'IMG', 'Name of the directory containing the raw images')
flags.DEFINE_string('label_dir', 'voc-labels', 'Name of the directory containing the raw images')

FLAGS = flags.FLAGS


def join_datasets(input_path, output_path, img_dir, label_dir):
    uid = 1

    dirs = get_immediate_subdirectories(input_path)

    # create new structure in output path
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    out_img_path = join(output_path, img_dir)

    if not os.path.exists(out_img_path):
        os.mkdir(out_img_path)

    out_voc_path = join(output_path, label_dir)

    if not os.path.exists(out_voc_path):
        os.mkdir(out_voc_path)

    # for each dataset inside the root directory
    for dataset in dirs:
        path = join(input_path, dataset)
        in_voc_path = join(path, label_dir)

        annotations = glob.glob(in_voc_path + '/*.xml')

        for annotation in annotations:
            tree = ET.parse(annotation)
            root = tree.getroot()

            in_img_path = root.find('path').text

            if not os.path.isfile(in_img_path):
                raise Exception('{} does not exist'.format(in_img_path))

            out_img_path_full = join(out_img_path, 'image{uid}.jpg'.format(uid=uid))
            out_voc_path_full = join(out_voc_path, 'image{uid}.xml'.format(uid=uid))

            copy_img_label(in_img_path, out_img_path_full, annotation, out_voc_path_full)

            uid += 1


def main(_):
    join_datasets(input_path=FLAGS.input_path,
                  output_path=FLAGS.output_path,
                  img_dir=FLAGS.img_dir,
                  label_dir=FLAGS.label_dir)


if __name__ == '__main__':
    tf.app.run()
