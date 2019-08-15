from utils.tl_utils import copy_img_label, get_image_path_from_label
import os
import tensorflow as tf
import glob


flags = tf.app.flags
flags.DEFINE_string('src', '', 'Source directory')
flags.DEFINE_string('dst', '', 'Destination')
flags.DEFINE_string('label_subdir', 'labels', 'Subdirectoy containing the labels')
FLAGS = flags.FLAGS


def main(_):
    label_src = os.path.join(FLAGS.src, FLAGS.label_subdir)
    img_dst = os.path.join(FLAGS.dst, 'img')
    label_dst = os.path.join(FLAGS.dst, 'labels')

    if not os.path.exists(FLAGS.src):
        raise Exception('Directory does not exist: {0}'.format(FLAGS.src))

    if not os.path.exists(label_src):
        raise Exception('Directory does not exist: {0}'.format(label_src))

    if not os.path.exists(FLAGS.dst):
        os.mkdir(FLAGS.dst)

    if not os.path.exists(img_dst):
        os.mkdir(img_dst)

    if not os.path.exists(label_dst):
        os.mkdir(label_dst)

    labels = glob.glob(os.path.join(label_src, '*.xml'))

    for label in labels:
        img_src = get_image_path_from_label(label)
        name = img_src.split('/')[-1].split('.')[0]

        copy_img_label(img_src=img_src,
                       img_dst=os.path.join(img_dst, '{name}.jpg'.format(name=name)),
                       label_src=label,
                       label_dst=os.path.join(label_dst, '{name}.xml'.format(name=name)))


if __name__ == '__main__':
    tf.app.run()
