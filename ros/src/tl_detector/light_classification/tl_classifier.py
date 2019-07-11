import numpy as np
import tensorflow as tf
from styx_msgs.msg import TrafficLight
from utils import label_map_util
from utils import ops as utils_ops
import cv2
from collections import Counter


class TLClassifier(object):
    def __init__(self, model_path, label_map_path, min_score=0.75):
        self.detection_graph = tf.Graph()
        self.min_score = min_score

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

        self.category_index = label_map_util.create_category_index_from_labelmap(label_map_path, use_display_name=False)

        self.dict = \
            {
                'red': TrafficLight.RED,
                'yellow': TrafficLight.YELLOW,
                'green': TrafficLight.GREEN,
                'unknown': TrafficLight.UNKNOWN
            }

    def get_classification(self, img):

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_expanded = np.expand_dims(img_rgb, axis=0)

        with self.detection_graph.as_default():
            (boxes, scores, classes, num) = self.sess.run(
                [self.d_boxes, self.d_scores,
                 self.d_classes, self.num_d],
                feed_dict={self.image_tensor: img_expanded})

        # boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        classes = np.squeeze(classes).astype(np.int32)

        detected_lights = [i[0] for i in zip(classes, scores) if i[1] > self.min_score]
        votes = Counter(detected_lights)
        dict = {}

        for value in votes.values():
            dict[value] = []

            for (key, value) in votes.iteritems():
                dict[value].append(key)

        max_vote = sorted(dict.keys(), reverse=True)[0]

        # class_name = 'unknown'
        # for i in range(boxes.shape[0]):
        #     if scores[i] > self.min_score:
        #         # if classes[i] in self.category_index.keys():
        #         class_name = self.category_index[classes[i]]['name']

        return self.dict[self.category_index[dict[max_vote][0]]['name']]

    @staticmethod
    def run_inference_for_single_image(image, graph):
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

    @staticmethod
    def load_image_into_numpy_array(image):
        try:
            (im_width, im_height, _) = image.shape
        except AttributeError:
            image = np.array(image)
            (im_width, im_height, _) = image.shape

        return np.array(image).reshape((im_height, im_width, 3)).astype(np.uint8)