import tensorflow as tf
import numpy as np
import glob
from PIL import Image
from utils import visualization_utils as vis_utils
from utils import ops as utils_ops
from utils import label_map_util
from utils import tl_utils
import os



flags = tf.app.flags
flags.DEFINE_string('model',
                    '/home/basti/tools/models/research/object_detection/training/fine_tuned_model/frozen_inference_graph.pb',
                    'Path pointing to the frozen inference graph (.pb)')

flags.DEFINE_string('label_map',
                    '/home/basti/tools/models/research/object_detection/training/tl_label_map.pbtxt',
                    'Path pointing to the label map (.pbtxt)')

flags.DEFINE_string('img_path',
                    '/home/basti/Udacity/CarND-Capstone/sim_datasets/raw/tl-set-1/IMG',
                    'Directory containing images')

flags.DEFINE_string('output_path',
                    '/home/basti/Udacity/CarND-Capstone/sim_datasets/raw/tl-set-1/classification_result',
                    'Directory to store the visualized classification result.')

FLAGS = flags.FLAGS


class TrafficLightClassifier(object):
    def __init__(self, model, label_map, img_path, output_path):

        self.model = model
        self.label_map = label_map
        self.img_path = img_path
        self.output_path = output_path

        self.detection_graph = tf.Graph()

        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.model, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

                self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
                self.d_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
                self.d_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
                self.d_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
                self.num_d = self.detection_graph.get_tensor_by_name('num_detections:0')
                self.sess = tf.Session(graph=self.detection_graph)

        self.category_index = label_map_util.create_category_index_from_labelmap(self.label_map, use_display_name=True)

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
        (im_width, im_height) = image.size

        return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

    def run(self):
        samples = glob.glob(self.img_path + '/*.jpg')
        num_lines = len(samples)
        uid = 1
        i = 0

        for sample in samples:
            tl_utils.print_progress(i, num_lines, prefix='Progress', suffix='Complete', bar_length=50)
            img = Image.open(sample)
            img = TrafficLightClassifier.load_image_into_numpy_array(img)
            img_expanded = np.expand_dims(img, axis=0)

            output_dict = TrafficLightClassifier.run_inference_for_single_image(img_expanded, self.detection_graph)
            boxes = output_dict['detection_boxes']
            classes = output_dict['detection_classes']
            scores = output_dict['detection_scores']

            out_img = vis_utils.visualize_boxes_and_labels_on_image_array(image=img, boxes=boxes, classes=classes,
                                                                          scores=scores,
                                                                          category_index=self.category_index,
                                                                          use_normalized_coordinates=True,
                                                                          line_thickness=5)

            vis_utils.save_image_array_as_png(out_img, os.path.join(self.output_path, 'image{}.png'.format(uid)))
            uid += 1
            i += 1


def main(_):
    classifier = TrafficLightClassifier(model=FLAGS.model,
                                        label_map=FLAGS.label_map,
                                        img_path=FLAGS.img_path,
                                        output_path=FLAGS.output_path)
    classifier.run()


if __name__ == '__main__':
    tf.app.run()
