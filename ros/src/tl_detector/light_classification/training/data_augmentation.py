import random
import skimage as sk
from skimage import transform, util, io
import os
import tensorflow as tf
import numpy as np
import cv2
import xml.etree.ElementTree as ET
from utils.tl_utils import convert_to_pascal_voc, copy_img_label, get_image_path_from_label
import csv


flags = tf.app.flags
flags.DEFINE_string('output_path', '', 'Path to save augmented images')
flags.DEFINE_string('label_path', '', 'Directory containing labels')
FLAGS = flags.FLAGS


def _read_in_data(label):
    tree = ET.parse(label)
    root = tree.getroot()

    # read in image
    img_path = root.find('path').text

    img = sk.io.imread(img_path)
    img = np.array(img)

    # read in bounding boxes
    obj = root.find('object')

    if obj is None:
        return None, None, None

    bboxes = []
    for elem in obj.findall('bndbox'):
        x_min = np.float32(elem.find('xmin').text)
        y_min = np.float32(elem.find('ymin').text)
        x_max = np.float32(elem.find('xmax').text)
        y_max = np.float32(elem.find('ymax').text)

        bbox = np.array([y_min, x_min, y_max, x_max])
        bboxes.append(bbox)

    l = obj.find('name').text

    return img, bboxes, l


def _save_augmented(img_path, img_name, label_path, img, label, bboxes):

    if len(bboxes) > 0:
        # save image
        io.imsave(os.path.join(img_path, img_name), img)

        # save label
        bboxes = np.array(bboxes).astype(np.float32)
        bboxes[:, [0, 2]] /= img.shape[0]
        bboxes[:, [1, 3]] /= img.shape[1]

        if len(img.shape) == 3:
            shape = img.shape
        elif len(img.shape) == 2:
            shape = [img.shape[0], img.shape[1], 1]
        else:
            raise Exception('Unknown image shape: {0}'.format(img.shape))

        convert_to_pascal_voc(label_path, img_path, img_name, bboxes, label, shape)
        return 1
    return 0


def random_rotation(img, bboxes):
    raise Exception('Not Implemented')
    # pick a random degree of rotation between 25% on the left and 25% on the right
    random_degree = random.uniform(-25, 25)
    return sk.transform.rotate(img, random_degree), bboxes


def random_noise(img, boxes):
    # add random noise to the image
    return sk.util.random_noise(img), boxes


def horizontal_flip(img, bboxes):
    # horizontal flip doesn't need skimage, it's easy as flipping the image array of pixels !
    img_center = img.shape[1]//2

    tboxes = bboxes

    for tbox in tboxes:
        tbox[[1, 3]] += 2*(img_center - tbox[[1, 3]])
        box_w = abs(tbox[1] - tbox[3])
        tbox[1] -= box_w
        tbox[3] += box_w

    if len(img.shape) == 3:
        flipped = img[:, ::-1, :]
    elif len(img.shape) == 2:
        flipped = img[:, ::-1]
    else:
        raise Exception('Unknown image shape: {0}'.format(img.shape))

    return flipped, tboxes


def blur(img, bboxes):
    return cv2.GaussianBlur(img, (5, 5), 0), bboxes


def sharpen(img, bboxes):
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    return cv2.filter2D(img, -1, kernel), bboxes


def random_image_scale(img, bboxes, mn=1.2, mx=1.7):
    height = img.shape[0]
    width = img.shape[1]
    f = random.uniform(mn, mx)

    h_offset = 200
    v_offset = 200

    # scale image
    img_scaled = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)

    while h_offset + height > img_scaled.shape[0]:
        h_offset -= 1

    while v_offset + width > img_scaled.shape[1]:
        v_offset -= 1

    # crop to original shape
    img_cropped = img_scaled[v_offset:v_offset+height, h_offset:h_offset+width]

    # scale and translate bounding boxes
    bboxes = np.array(bboxes).astype(np.float32) * f
    bboxes[:, [1, 3]] -= h_offset
    bboxes[:, [0, 2]] -= v_offset

    # check if bounding boxes are still within the scaled image
    out_bboxes = []
    for i, box in enumerate(bboxes):
        box[0] = max(min(box[0], height), 0)
        box[1] = max(min(box[1], width), 0)
        box[2] = max(min(box[2], height), 0)
        box[3] = max(min(box[3], width), 0)

        b_h = abs(box[3] - box[1])
        b_w = abs(box[2]-box[0])

        if 50 < b_w < b_h:
            out_bboxes.append(box)

    return img_cropped, out_bboxes


available_transformations = {
    # 'rotate': random_rotation,
    # 'noise': random_noise,
    'horizontal_flip': horizontal_flip,
    # 'blur': blur,
    # 'sharpen': sharpen,
    'scale': random_image_scale,
}


def _write_csv(old, new):
    with open('/home/basti/Desktop/rosbag_images/augmented/mapping.csv', mode='a') as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow([old, new])


def main(_):
    label_path = FLAGS.label_path
    output_path = FLAGS.output_path
    out_img_path = os.path.join(output_path, 'img')
    out_label_path = os.path.join(output_path, 'labels')

    if not os.path.exists(out_img_path):
        os.makedirs(out_img_path)

    if not os.path.exists(out_label_path):
        os.makedirs(out_label_path)

    labels = [os.path.join(label_path, f) for f in os.listdir(label_path) if
              os.path.isfile(os.path.join(label_path, f))]

    i = 0
    for label in labels:

        img, bboxes, l = _read_in_data(label)

        if img is None or bboxes is None or l is None:
            continue

        name = 'image{0}'.format(i)
        img_path = get_image_path_from_label(label)

        copy_img_label(img_src=img_path,
                       img_dst=os.path.join(out_img_path, '{name}.jpg'.format(name=name)),
                       label_src=label,
                       label_dst=os.path.join(out_label_path, '{name}.xml').format(name=name))

        i += 1

        transformed_image = img
        transformed_boxes = bboxes

        # random transformation for a single image
        key = random.choice(list(available_transformations))
        transformed_image, transformed_boxes = available_transformations[key](transformed_image, transformed_boxes)

        name = 'image{0}.jpg'.format(i)

        ans = _save_augmented(img_path=out_img_path,
                              img_name=name,
                              label_path=out_label_path,
                              img=transformed_image,
                              label=l,
                              bboxes=transformed_boxes)

        if ans:
            i += 1
            # _write_csv(old=img_path.split('/')[-1], new=name)


if __name__ == '__main__':
    tf.app.run()
