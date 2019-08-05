import random
import skimage as sk
from skimage import transform, util, io
import os
import tensorflow as tf
import numpy as np
import cv2
import xml.etree.ElementTree as ET
from utils.tl_utils import convert_to_pascal_voc


flags = tf.app.flags
flags.DEFINE_string('image_path', '/home/basti/Desktop/rosbags/rosbag-1/exported', 'Directory containing images')
flags.DEFINE_string('output_path', '/home/basti/Desktop/rosbags/rosbag-1/augmented', 'Path to save augmented images')
flags.DEFINE_string('label_path', '/home/basti/Desktop/rosbags/rosbag-1/labels', 'Directory containing labels')
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

    bboxes = []
    for elem in obj.findall('bndbox'):
        x_min = np.float32(elem[0].text)
        y_min = np.float32(elem[1].text)
        x_max = np.float32(elem[2].text)
        y_max = np.float32(elem[3].text)

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
        convert_to_pascal_voc(label_path, img_path, img_name, bboxes, label, img.shape)
        return 1
    return 0


def random_rotation(img, bboxes):
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

    return img[:, ::-1, :], tboxes


def blur(img, bboxes):
    return cv2.GaussianBlur(img, (5, 5), 0), bboxes


def sharpen(img, bboxes):
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    return cv2.filter2D(img, -1, kernel), bboxes


def random_image_scale(img, bboxes):

    h = 600
    w = 800
    f = 1.5

    h_t = 200
    v_t = 100

    img_scaled = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
    img_cropped = img_scaled[v_t:v_t+h, h_t:h_t+w]

    out_bboxes = np.array(bboxes).astype(np.float32) * f
    out_bboxes[:, [1, 3]] -= h_t
    out_bboxes[:, [0, 2]] -= v_t

    rm = []
    i = 0
    for box in out_bboxes:
        box[0] = min(box[0], 600.)
        box[1] = min(box[1], 800.)
        box[2] = min(box[2], 600.)
        box[3] = min(box[3], 800.)

        if abs(box[3] - box[1]) < 100.:
            rm.append(i)
            print('aaaaaaaaaaaaaaaaaaaaaaaaaa')

        i += 1

    out_bboxes = np.delete(out_bboxes, rm)

    return img_cropped, [out_bboxes]


available_transformations = {
    # 'rotate': random_rotation,
    #'noise': random_noise,
    #'horizontal_flip': horizontal_flip,
    #'blur': blur,
    #'sharpen': sharpen,
    'scale': random_image_scale,
}


def main(_):
    label_path = FLAGS.label_path
    output_path = FLAGS.output_path
    out_img_path = os.path.join(output_path, 'img')
    out_label_path = os.path.join(output_path, 'label')

    if not os.path.exists(out_img_path):
        os.mkdir(out_img_path)

    if not os.path.exists(out_label_path):
        os.mkdir(out_label_path)

    num_files_desired = 50

    labels = [os.path.join(label_path, f) for f in os.listdir(label_path) if
              os.path.isfile(os.path.join(label_path, f))]

    num_generated_files = 0

    while num_generated_files < num_files_desired:
        # pick random image
        label = random.choice(labels)

        img, bboxes, l = _read_in_data(label)

        num_transformations_to_apply = random.randint(1, len(available_transformations))

        num_transformations = 0
        transformed_image = img
        transformed_boxes = bboxes

        # random transformation to apply for a single image
        key = random.choice(list(available_transformations))
        transformed_image, transformed_boxes = available_transformations[key](transformed_image, transformed_boxes)

        img_name = 'augmented_image_{}.jpg'.format(num_generated_files)

        ans = _save_augmented(img_path=out_img_path,
                              img_name=img_name,
                              label_path=out_label_path,
                              img=transformed_image,
                              label=l,
                              bboxes=transformed_boxes)

        if ans:
            num_generated_files += 1


if __name__ == '__main__':
    tf.app.run()
