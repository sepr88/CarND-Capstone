# -*- coding: utf-8 -*-

import cv2
from os.path import join
from styx_msgs.msg import TrafficLight
import sys
from lxml import etree as ET
import numpy as np
import os
from shutil import copyfile

StringToState = \
    {
        'red': TrafficLight.RED,
        'yellow': TrafficLight.YELLOW,
        'green': TrafficLight.GREEN,
        'unknown': TrafficLight.UNKNOWN
    }

StateToString = \
    {
        TrafficLight.RED: 'red',
        TrafficLight.YELLOW: 'yellow',
        TrafficLight.GREEN: 'green',
        TrafficLight.UNKNOWN: 'unknown'
    }


def save_td(uid, cv_image, label, csv_path, img_path):
    """
    Save image and label
    @params:
        uid         - Required : unique identifier, used as image name (Var)
        cv_image:   - Required : the image to be saved (Image)
        label:      - Required : the associated label, e.g. light.state (Str)
        path:       - Required : folder to store the img data (Str)
    """

    img_path_full = join(img_path, 'image{uid}.jpg'.format(uid=uid))

    # save image
    cv2.imwrite(img_path_full, cv_image)

    # save labels (append label to csv
    with open(join(csv_path, 'labels.csv'), 'a') as f:
        f.write('{img_path_full};{label}\n'.format(img_path_full=img_path_full, label=label))
                        
    return True


def tl_state_to_label(state):
    if state == TrafficLight.RED:
        return 'red'
    
    if state == TrafficLight.GREEN:
        return 'green'
    
    if state == TrafficLight.YELLOW:
        return 'yellow'
    
    return 'off'


# Print iterations progress
def print_progress(iteration, total, prefix='', suffix='', decimals=1, bar_length=100):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        bar_length  - Optional  : character length of bar (Int)
    """
    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = '█' * filled_length + '-' * (bar_length - filled_length)

    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),

    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()


def yes_or_no(question):
    reply = str(raw_input(question + ' (y/n): ')).lower().strip()
    if reply[0] == 'y':
        return True
    if reply[0] == 'n':
        return False
    else:
        return yes_or_no(question)


def convert_to_pascal_voc(out_path, img_path, img_name, boxes, label, img_shape):

    annotation = ET.Element("annotation")

    ET.SubElement(annotation, "folder").text = img_path.split('/')[-1]
    ET.SubElement(annotation, "filename").text = img_name

    ET.SubElement(annotation, "path").text = str(os.path.join(img_path, img_name))

    source = ET.SubElement(annotation, "source")
    ET.SubElement(source, "database").text = "Unknown"

    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(img_shape[1])
    ET.SubElement(size, "height").text = str(img_shape[0])
    ET.SubElement(size, "depth").text = str(img_shape[2])
    ET.SubElement(annotation, "segmented").text = "0"

    if boxes is not None:
        for box in boxes:
            obj_elem = ET.SubElement(annotation, "object")

            ET.SubElement(obj_elem, "name").text = str(label)
            ET.SubElement(obj_elem, "pose").text = "Unspecified"
            ET.SubElement(obj_elem, "truncated").text = "0"
            ET.SubElement(obj_elem, "difficult").text = "0"

            bndbox = ET.SubElement(obj_elem, "bndbox")

            ET.SubElement(bndbox, "xmin").text = str(np.uint32(box[1] * img_shape[1]))
            ET.SubElement(bndbox, "ymin").text = str(np.uint32(box[0] * img_shape[0]))
            ET.SubElement(bndbox, "xmax").text = str(np.uint32(box[3] * img_shape[1]))
            ET.SubElement(bndbox, "ymax").text = str(np.uint32(box[2] * img_shape[0]))

    tree = ET.ElementTree(annotation)

    name = img_name.split('.')[0]

    tree.write(os.path.join(out_path, '{}.xml'.format(name)), pretty_print=True)


def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]


def copy_img_label(img_src, img_dst, label_src, label_dst, im_shape=None):

    # copy files
    try:
        copyfile(img_src, img_dst)
    except:
        pass

    copyfile(label_src, label_dst)

    # update path in label[.xml] file
    tree = ET.parse(label_dst, ET.XMLParser(remove_blank_text=True))
    root = tree.getroot()
    elem = root.find('path')
    elem.text = img_dst

    if im_shape is not None:
        elem = root.find('size')
        elem.find('height').text = str(im_shape[0])
        elem.find('width').text = str(im_shape[1])
        elem.find('depth').text = str(im_shape[2])

    img_name = img_dst.split('/')[-1]
    elem = root.find('filename')
    elem.text = img_name

    obj = root.find('object')

    if obj is not None:
        for bbox in obj.findall('bndbox'):
            xmin_obj = bbox.find('xmin')
            ymin_obj = bbox.find('ymin')
            xmax_obj = bbox.find('xmax')
            ymax_obj = bbox.find('ymax')

            xmin_val = np.uint32(xmin_obj.text)
            ymin_val = np.uint32(ymin_obj.text)
            xmax_val = np.uint32(xmax_obj.text)
            ymax_val = np.uint32(ymax_obj.text)

            bbox.remove(xmin_obj)
            bbox.remove(ymin_obj)
            bbox.remove(xmax_obj)
            bbox.remove(ymax_obj)

            ET.SubElement(bbox, "xmin").text = str(xmin_val)
            ET.SubElement(bbox, "ymin").text = str(ymin_val)
            ET.SubElement(bbox, "xmax").text = str(xmax_val)
            ET.SubElement(bbox, "ymax").text = str(ymax_val)

    tree.write(label_dst, pretty_print=True)


def get_image_path_from_label(label):
    if not os.path.isfile(label):
        raise Exception('File does not exist: {0}'.format(label))

    tree = ET.parse(label)
    root = tree.getroot()

    return root.find('path').text
