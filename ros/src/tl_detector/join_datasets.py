import sys
import os
import csv
from os.path import join
from shutil import copyfile
import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_string('input_path', '', 'Root directory containing all datasets to be joined.')
flags.DEFINE_string('output_path', '', 'Root directory to store the new dataset')
FLAGS = flags.FLAGS


def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]


def join_datasets(input_path, output_path):
    uid = 1

    dirs = get_immediate_subdirectories(input_path)

    # create new structure in output path
    out_img_path = join(output_path, 'IMG')

    if not os.path.exists(out_img_path):
        os.mkdir(out_img_path)

    # for each dataset inside the root directory
    for dataset in dirs:
        path = join(input_path, dataset)
        in_img_path = join(path, 'IMG')

        # open csv
        with open(join(path, 'labels.csv')) as in_csv:
            with open(join(output_path, 'labels.csv'), 'a') as out_csv:
                r = csv.reader(in_csv, delimiter=';')

                for line in r:
                    in_img_name = line[0].split('/')[-1]
                    file_ext = in_img_name.split('.')[-1]
                    out_img_name = 'image{uid}.{ext}'.format(uid=uid, ext=file_ext)
                    out_csv.write('{img_path};{label}\n'.format(img_path=join(out_img_path, out_img_name),
                                                                label=line[1]))

                    # copy and rename each images to the output path
                    copyfile(join(in_img_path, in_img_name), join(out_img_path, out_img_name))
                    uid += 1

    return True


def main(_):
    join_datasets(input_path=FLAGS.input_path,
                  output_path=FLAGS.output_path)


if __name__ == '__main__':
    tf.app.run()
