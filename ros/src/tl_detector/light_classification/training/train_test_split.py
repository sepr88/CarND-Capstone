import os
from utils.tl_utils import copy_img_label
from sklearn.model_selection import train_test_split
import glob
import xml.etree.ElementTree as ET
import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_string('input_path', '', 'Root directory to dataset.')
flags.DEFINE_string('output_path', '', 'Root directory to output datasets.')
flags.DEFINE_float('validation_size', 0.2, 'Root directory to output datasets.')
flags.DEFINE_string('img_dir', 'IMG', '')
flags.DEFINE_string('label_dir', 'voc-labels', '')
FLAGS = flags.FLAGS


def _mkdirs(dirs):
    for dir in dirs:
        if not os.path.exists(dir):
            os.makedirs(dir)


def split_dataset(in_path, out_path, img_dir, label_dir, validation_size):

    if not os.path.exists(in_path):
        raise Exception("input path does not exist: {}".format(in_path))

    in_img_path = os.path.join(in_path, img_dir)
    if not os.path.exists(in_img_path):
        raise Exception("image path does not exist: {}".format(in_img_path))

    in_label_path = os.path.join(in_path, label_dir)
    if not os.path.exists(in_label_path):
        raise Exception("label path does not exist: {}".format(in_label_path))

    train_base_path = os.path.join(out_path, 'train_set')
    train_img_path = os.path.join(train_base_path, 'img')
    train_label_path = os.path.join(train_base_path, 'labels')

    valid_base_path = os.path.join(out_path, 'validation_set')
    valid_img_path = os.path.join(valid_base_path, 'img')
    valid_label_path = os.path.join(valid_base_path, 'labels')

    _mkdirs([out_path, train_base_path, train_img_path, train_label_path,
             valid_base_path, valid_img_path, valid_label_path])

    # read in training data
    labels_train, labels_test = train_test_split(glob.glob(in_label_path + '/*.xml'),
                                                 test_size=validation_size,
                                                 shuffle=True)

    # save training data
    for label in labels_train:
        tree = ET.parse(os.path.join(in_label_path, label))
        root = tree.getroot()
        elem = root.find('filename')
        image = elem.text

        if not image.endswith('.jpg'):
            image += '.jpg'

        label_name = label.split('/')[-1]

        copy_img_label(img_src=os.path.join(in_img_path, image),
                       img_dst=os.path.join(train_img_path, image),
                       label_src=label,
                       label_dst=os.path.join(train_label_path, label_name))

    # save validation data
    for label in labels_test:
        tree = ET.parse(os.path.join(in_label_path, label))
        root = tree.getroot()
        elem = root.find('filename')
        image = elem.text

        if not image.endswith('.jpg'):
            image += '.jpg'

        label_name = label.split('/')[-1]

        copy_img_label(img_src=os.path.join(in_img_path, image),
                       img_dst=os.path.join(valid_img_path, image),
                       label_src=label,
                       label_dst=os.path.join(valid_label_path, label_name))


def main(_):
    split_dataset(in_path=FLAGS.input_path,
                  out_path=FLAGS.output_path,
                  img_dir=FLAGS.img_dir,
                  label_dir=FLAGS.label_dir,
                  validation_size=FLAGS.validation_size)


if __name__ == '__main__':
    tf.app.run()
