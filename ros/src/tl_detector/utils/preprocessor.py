import cv2
import tensorflow as tf
import os
import glob
from utils.tl_utils import copy_img_label, get_image_path_from_label

flags = tf.app.flags
flags.DEFINE_string('label_path', '', 'Directory containing the labels')
flags.DEFINE_string('output_path', '', 'Path to save augmented images')
FLAGS = flags.FLAGS


def normalize_image(img):
    cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)
    return img


def crop_image(img):
    return img[0:400, :]


def convert_to_grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def run_preprocessor_pipeline(img):
    # img = normalize_image(img)
    img = crop_image(img)
    # img = convert_to_grayscale(img)
    return img


def _create_dirs(dirs):
    for dir in dirs:
        if not os.path.exists(dir):
            os.mkdir(dir)


def _prepare_dirs(output_path, label_path):

    out_img_path = os.path.join(output_path, 'img')
    out_label_path = os.path.join(output_path, 'labels')

    _create_dirs([output_path, out_img_path, out_label_path])

    if not os.path.exists(label_path):
        raise Exception('Directory does not exist: {0}'.format(label_path))

    return glob.glob(os.path.join(label_path, '*.xml')), out_img_path, out_label_path


def _read_image(label):
    img_path = get_image_path_from_label(label)
    img = cv2.imread(img_path)
    return img


def _write_img_and_label(img_path, img, label_src, label_dst):
    cv2.imwrite(img_path, img)

    if len(img.shape) == 3:
        im_shape = img.shape
    elif len(img.shape) == 2:
        im_shape = [img.shape[0], img.shape[1], 1]
    else:
        raise Exception('Unknown image shape: {0}'.format(img.shape))

    copy_img_label(img_src=img_path, img_dst=img_path, label_src=label_src, label_dst=label_dst, im_shape=im_shape)


def main(_):

    label_list, out_img_path, out_label_path = _prepare_dirs(output_path=FLAGS.output_path,
                                                             label_path=FLAGS.label_path)

    for i, label in enumerate(label_list):
        img = _read_image(label)
        img = run_preprocessor_pipeline(img)

        name = 'image{0}'.format(i)
        _write_img_and_label(img_path=os.path.join(out_img_path, '{name}.jpg'.format(name=name)),
                             img=img,
                             label_src=label,
                             label_dst=os.path.join(out_label_path, '{name}.xml'.format(name=name)))


if __name__ == '__main__':
    tf.app.run()
