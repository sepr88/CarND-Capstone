import os
import csv
import sys
import cv2
import numpy as np
import tensorflow as tf
import xml.etree.cElementTree as ET
import visualization_utils as vis_utils
import label_map_util
from PIL import Image
from utils import ops as utils_ops
import tl_utils
from time import sleep

PATH_TO_LABELS = '/home/basti/Udacity/CarND-Capstone/ros/src/tl_detector/data/mscoco_label_map.pbtxt'
PATH_TO_IMG = '/home/basti/Udacity/CarND-Capstone/ros/src/tl_detector/examples/'
PATH_TO_MODEL = '/home/basti/Udacity/CarND-Capstone/ros/src/tl_detector/rfcn_resnet101_coco_2018_01_28/frozen_inference_graph.pb'

class TrafficLightClassifier(object):
    def __init__(self, base_path):

        self.detection_graph = tf.Graph()

        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_MODEL, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

                self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
                self.d_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
                self.d_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
                self.d_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
                self.num_d = self.detection_graph.get_tensor_by_name('num_detections:0')
                self.sess = tf.Session(graph=self.detection_graph)

        self.category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

        self.uid = 1
        self.base_path = base_path
        self.img_path = os.path.join(self.base_path, 'IMG')
        self.voc_path = os.path.join(self.base_path, 'voc-labels')
        self.csv_path = os.path.join(self.base_path, 'labels.csv')
        self.overwrite = True

        # create new output path if doesnt exist
        if not os.path.exists(self.voc_path):
            os.mkdir(self.voc_path)
        else:
            self.overwrite = tl_utils.yes_or_no('Overwrite existing?')

        pass

    def get_classification(self, img):
        # Bounding Box Detection.
        with self.detection_graph.as_default():
            # Expand dimension since the model expects image to have shape [1, None, None, 3].
            (boxes, scores, classes, num) = self.sess.run([self.d_boxes, self.d_scores, self.d_classes, self.num_d],
                                                          feed_dict={self.image_tensor: img})
        return boxes, scores, classes, num

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
                tl_utils.print_progress(i, num_lines, prefix='Progress', suffix='Complete', bar_length=50)
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
                img = self.load_image_into_numpy_array(img)

                self.process_image(img=img, label=label, img_name=img_name, img_path_full=img_path_full)
                i += 1

    def process_image(self, img, label, img_name, img_path_full):

        img_expanded = np.expand_dims(img, axis=0)

        output_dict = self.run_inference_for_single_image(img_expanded, self.detection_graph)
        boxes = output_dict['detection_boxes']
        classes = output_dict['detection_classes']
        scores = output_dict['detection_scores']


        '''
        boxes, scores, classes, num = self.get_classification(img_expanded)
        boxes = np.squeeze(boxes)
        classes = np.squeeze(classes).astype(np.uint8)
        scores = np.squeeze(scores)
        '''

        '''
        out_img = vis_utils.visualize_boxes_and_labels_on_image_array(image=img, boxes=boxes, classes=classes,
                                                                      scores=scores, category_index=self.category_index,
                                                                      use_normalized_coordinates=True,
                                                                      line_thickness=10)

        vis_utils.save_image_array_as_png(out_img, os.path.join(PATH_TO_IMG, 'image{}.png'.format(self.uid)))
        '''

        # select only boxes labeled "Traffic Light" with a score of at least 90%
        traffic_lights = np.array([b[0] for b in zip(boxes, classes, scores) if b[1] == 10 and b[2] > 0.9])

        # save
        convert_to_pascal_voc(out_path=self.voc_path, img_path=self.img_path, img_name=img_name, boxes=traffic_lights,
                              label=label, img_shape=img.shape)

        self.uid += 1
        return True

    def load_image_into_numpy_array(self, image):
        (im_width, im_height) = image.size
        return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

    def run_inference_for_single_image(self, image, graph):
        with graph.as_default():
            with tf.Session() as sess:
                # Get handles to input and output tensors
                ops = tf.get_default_graph().get_operations()
                all_tensor_names = {output.name for op in ops for output in op.outputs}
                tensor_dict = {}

                for key in ['num_detections', 'detection_boxes', 'detection_scores', 'detection_classes',
                            'detection_masks']:
                    tensor_name = key + ':0'
                    if tensor_name in all_tensor_names:
                        tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)

                if 'detection_masks' in tensor_dict:
                    # The following processing is only for single image
                    detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                    detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])

                    real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                    detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                    detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(detection_masks,
                                                                                          detection_boxes,
                                                                                          image.shape[1],
                                                                                          image.shape[2])

                    detection_masks_reframed = tf.cast(tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                    # Follow the convention by adding back the batch dimension
                    tensor_dict['detection_masks'] = tf.expand_dims(detection_masks_reframed, 0)
                image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

                # Run inference
                output_dict = sess.run(tensor_dict, feed_dict={image_tensor: image})

                # all outputs are float32 numpy arrays, so convert types as appropriate
                output_dict['num_detections'] = int(output_dict['num_detections'][0])
                output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.int64)
                output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
                output_dict['detection_scores'] = output_dict['detection_scores'][0]

                if 'detection_masks' in output_dict:
                    output_dict['detection_masks'] = output_dict['detection_masks'][0]

        return output_dict

    def label_exists(self, img_name):
        xml_file = img_name.split('.')[0]
        xml_file = os.path.join(self.voc_path, '{}.xml'.format(xml_file))
        return os.path.isfile(xml_file)


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

    for box in boxes:
        obj_elem = ET.SubElement(annotation, "object")

        ET.SubElement(obj_elem, "name").text = str(label)
        ET.SubElement(obj_elem, "pose").text = "Unspecified"
        ET.SubElement(obj_elem, "truncated").text = "0"
        ET.SubElement(obj_elem, "difficult").text = "0"

        bndbox = ET.SubElement(obj_elem, "bndbox")

        ET.SubElement(bndbox, "ymin").text = str(np.uint32(box[0] * img_shape[0]))
        ET.SubElement(bndbox, "xmin").text = str(np.uint32(box[1] * img_shape[1]))
        ET.SubElement(bndbox, "ymax").text = str(np.uint32(box[2] * img_shape[0]))
        ET.SubElement(bndbox, "xmax").text = str(np.uint32(box[3] * img_shape[1]))

    tree = ET.ElementTree(annotation)

    name = img_name.split('.')[0]

    tree.write(os.path.join(out_path, '{}.xml'.format(name)))

path = sys.argv[1]

classifier = TrafficLightClassifier(path)
classifier.autolabel_dataset()
