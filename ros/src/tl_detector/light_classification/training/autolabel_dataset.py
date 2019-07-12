import os
import csv
import numpy as np
import tensorflow as tf
from utils import label_map_util
from PIL import Image
from tensorflow import app
from light_classification.tl_classifier import TLClassifier
from utils.tl_utils import convert_to_pascal_voc, print_progress

MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir, 'data'))

flags = app.flags
flags.DEFINE_string('path', '', 'Root directory to dataset.')
flags.DEFINE_string('label_map_path', os.path.join(MODEL_PATH, 'mscoco_label_map.pbtxt'),
                    'Path to file containing the labels [.pbtxt]')
flags.DEFINE_string('model_path', os.path.join(MODEL_PATH, 'mscoco_frozen_inference_graph.pb'),
                    'Path to the frozen inference graph [.pb]')
flags.DEFINE_boolean('overwrite', True, 'Whether to overwrite existing annotations')

FLAGS = flags.FLAGS


class AutoLabel(object):
    def __init__(self, base_path, label_map_path, model_path, overwrite):

        self.detection_graph = tf.Graph()

        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(model_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

                self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
                self.d_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
                self.d_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
                self.d_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
                self.num_d = self.detection_graph.get_tensor_by_name('num_detections:0')
                self.sess = tf.Session(graph=self.detection_graph)

        self.category_index = label_map_util.create_category_index_from_labelmap(label_map_path, use_display_name=True)

        self.uid = 1
        self.base_path = base_path
        self.img_path = os.path.join(self.base_path, 'IMG')
        self.voc_path = os.path.join(self.base_path, 'voc-labels')
        self.csv_path = os.path.join(self.base_path, 'labels.csv')
        self.overwrite = overwrite

        # create new output path if doesnt exist
        if not os.path.exists(self.voc_path):
            os.mkdir(self.voc_path)

    def autolabel_dataset(self):

        if not os.path.exists(self.base_path):
            print('{} does not exist'.format(self.base_path))
            return

        if not os.path.exists(self.img_path):
            print('{} does not exist'.format(self.img_path))
            return

        if not os.path.isfile(self.csv_path):
            print('{} does not exist'.format(self.csv_path))
            return

        with open(self.csv_path, 'r') as f:
            reader = csv.reader(f, delimiter=';')

            num_lines = sum(1 for line in reader)
            f.seek(0)
            i = 0

            for line in reader:
                print_progress(i, num_lines, prefix='Progress', suffix='Complete', bar_length=50)
                img_name = line[0].split('/')[-1]
                img_path_full = os.path.join(self.img_path, img_name)
                label = line[1]

                if not os.path.isfile(img_path_full):
                    print('Skipping {}. File does not exist'.format(img_path_full))
                    i += 1
                    continue

                if not self.overwrite and self.label_exists(img_name=img_name):
                    i += 1
                    continue

                img = Image.open(img_path_full)
                img = AutoLabel.load_image_into_numpy_array(img)

                self.process_image(img=img, label=label, img_name=img_name)
                i += 1

    def process_image(self, img, label, img_name):

        img_expanded = np.expand_dims(img, axis=0)

        output_dict = TLClassifier.run_inference_for_single_image(img_expanded, self.detection_graph)
        boxes = output_dict['detection_boxes']
        classes = output_dict['detection_classes']
        scores = output_dict['detection_scores']

        # select only boxes labeled "Traffic Light" with a score of at least 90%
        traffic_lights = np.array([b[0] for b in zip(boxes, classes, scores) if b[1] == 10 and b[2] > 0.9])

        # save
        convert_to_pascal_voc(out_path=self.voc_path, img_path=self.img_path, img_name=img_name, boxes=traffic_lights,
                              label=label, img_shape=img.shape)

        self.uid += 1
        return True

    def label_exists(self, img_name):
        xml_file = img_name.split('.')[0]
        xml_file = os.path.join(self.voc_path, '{}.xml'.format(xml_file))
        return os.path.isfile(xml_file)

    @staticmethod
    def load_image_into_numpy_array(image):
            (im_width, im_height) = image.size
            return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)


def main(_):
    classifier = AutoLabel(base_path=FLAGS.path,
                           label_map_path=FLAGS.label_map_path,
                           model_path=FLAGS.model_path,
                           overwrite=FLAGS.overwrite)

    classifier.autolabel_dataset()


if __name__ == '__main__':
    tf.app.run()

