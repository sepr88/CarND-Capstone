import os
from tl_utils import yes_or_no
from sklearn.model_selection import train_test_split
import glob
from shutil import copyfile
import xml.etree.ElementTree as ET
import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_string('input_path', '', 'Root directory to dataset.')
flags.DEFINE_string('output_path', '', 'Root directory to output datasets.')
flags.DEFINE_float('validation_size', 0.2, 'Root directory to output datasets.')
FLAGS = flags.FLAGS


def split_dataset(in_path, out_path, validation_size):

    if not os.path.exists(in_path):
        print("input path does not exist: {}".format(in_path))
        return

    in_img_path = os.path.join(in_path, 'IMG')
    if not os.path.exists(in_img_path):
        print("image path does not exist: {}".format(in_img_path))
        return

    in_label_path = os.path.join(in_path, 'voc-labels')
    if not os.path.exists(in_label_path):
        print("label path does not exist: {}".format(in_label_path))
        return

    overwrite = True
    train_base_path = os.path.join(out_path, 'train_set')
    train_img_path = os.path.join(train_base_path, 'img')
    train_label_path = os.path.join(train_base_path, 'labels')

    valid_base_path = os.path.join(out_path, 'validation_set')
    valid_img_path = os.path.join(valid_base_path, 'img')
    valid_label_path = os.path.join(valid_base_path, 'labels')

    if not os.path.exists(out_path):
        os.mkdir(out_path)
        os.mkdir(train_base_path)
        os.mkdir(train_img_path)
        os.mkdir(train_label_path)
        os.mkdir(valid_base_path)
        os.mkdir(valid_img_path)
        os.mkdir(valid_label_path)
    else:
        overwrite = yes_or_no('Overwrite existing?')

    if not overwrite:
        print('Not implemented. Aborting.')
        return

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

        copy(img_src=os.path.join(in_img_path, image),
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

        copy(img_src=os.path.join(in_img_path, image),
             img_dst=os.path.join(valid_img_path, image),
             label_src=label,
             label_dst=os.path.join(valid_label_path, label_name))


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


def main(_):
    split_dataset(FLAGS.input_path, FLAGS.output_path, FLAGS.validation_size)


if __name__ == '__main__':
    tf.app.run()
