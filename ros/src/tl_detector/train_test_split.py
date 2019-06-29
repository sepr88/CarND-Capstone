import sys
import os
from tl_utils import yes_or_no
from sklearn.model_selection import train_test_split
import glob
from shutil import copyfile
import numpy as np
import xml.etree.ElementTree as ET


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
                                                 test_size=np.float32(validation_size),
                                                 shuffle=True)

    # save training data
    for label in labels_train:
        tree = ET.parse(os.path.join(in_label_path, label))
        root = tree.getroot()
        elem = root.find('filename')
        image = elem.text

        label_name = label.split('/')[-1]

        copyfile(os.path.join(in_img_path, image), os.path.join(train_img_path, image))
        copyfile(label, os.path.join(train_label_path, label_name))

    # save validation data
    for label in labels_test:
        tree = ET.parse(os.path.join(in_label_path, label))
        root = tree.getroot()
        elem = root.find('filename')
        image = elem.text

        label_name = label.split('/')[-1]

        copyfile(os.path.join(in_img_path, image), os.path.join(valid_img_path, image))
        copyfile(label, os.path.join(valid_label_path, label_name))


input_path = sys.argv[1]
output_path = sys.argv[2]
validation_size = sys.argv[3]

split_dataset(input_path, output_path, validation_size)