#!/usr/bin/env python
import rospy
from timeit import default_timer as timer
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import yaml
from scipy.spatial import KDTree
from utils import tl_utils
from utils.tl_utils import StateToString
import os
import tf
from light_classification.tl_classifier import TLClassifier

STATE_COUNT_THRESHOLD_SITE = 2
STATE_COUNT_THRESHOLD = 1
LOGGING_THROTTLE_FACTOR = 1
CAMERA_IMG_PROCESS_RATE = 0.2  # s
WAYPOINT_DIFFERENCE = 100

COLLECT_TD = False
TD_RATE = 5  # only save every i-th image
OUTPUT_DIRNAME = 'tl-set-1'

BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir))
MODEL_PATH = os.path.join(BASE_PATH, 'data')
OUTPUT_PATH = os.path.join(MODEL_PATH, OUTPUT_DIRNAME)

class TLDetector(object):
    def __init__(self):

        rospy.init_node('tl_detector')

        self.initialized = False
        self.pose_received = False
        self.waypoints_received = False
        self.image_received = False

        self.pose = None
        self.waypoints = None
        self.waypoints_2d = None
        self.waypoint_tree = None
        self.camera_image = None
        self.lights = []
        self.bridge = CvBridge()

        self.td_id = 1
        self.img_count = 0

        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        self.process_count = 0
        self.last_img_processed = -1
        self.process_rate = CAMERA_IMG_PROCESS_RATE

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        rospy.Subscriber('/image_color', Image, self.image_cb, queue_size=1)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)
        self.is_site = self.config["is_site"]

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)
        self.tl_detector_state_pub = rospy.Publisher('/tl_detector_state', Int32, queue_size=1)

        self.state_count_th = STATE_COUNT_THRESHOLD

        if self.is_site:
            self.state_count_th = STATE_COUNT_THRESHOLD_SITE

        # Choose classifier
        if not self.is_site:
            self.classifier = TLClassifier(model_path=os.path.join(MODEL_PATH, 'sim_frozen_inference_graph.pb'),
                                           label_map_path=os.path.join(MODEL_PATH, 'tl_label_map.pbtxt'))
        else:
            self.classifier = TLClassifier(model_path=os.path.join(MODEL_PATH, 'site_frozen_inference_graph.pb'),
                                           label_map_path=os.path.join(MODEL_PATH, 'tl_label_map.pbtxt'))

        self.td_img_path = os.path.join(OUTPUT_PATH, 'IMG')

        if COLLECT_TD:
            if not os.path.exists(OUTPUT_PATH):
                os.mkdir(OUTPUT_PATH)
            else:
                raise Exception('Directory {} already exists. Please choose a different name.'.format(OUTPUT_PATH))

            if not os.path.exists(self.td_img_path):
                os.mkdir(self.td_img_path)

        rospy.init_node('tl_detector')
        self.last_img_processed = 0
        self.listener = tf.TransformListener()
        self.initialized = True
        rospy.spin()

    def is_initialized(self):
        return self.pose_received and self.waypoints_received and self.image_received and self.initialized

    def pose_cb(self, msg):
        self.pose = msg
        self.pose_received = True

    def waypoints_cb(self, waypoints):

        rospy.logwarn("Waypoints received")
        self.waypoints = waypoints
        self.waypoints_received = True

        if not self.waypoints_2d:
            self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints]
            self.waypoint_tree = KDTree(self.waypoints_2d)

    def traffic_cb(self, msg):
        self.lights = msg.lights

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        self.image_received = True

        is_initialized = self.is_initialized()

        self.tl_detector_state_pub.publish(Int32(is_initialized))

        if not is_initialized:
            return

        time_elapsed = timer() - self.last_img_processed

        # Do not process the camera image unless some time has passed from last processing
        if time_elapsed < self.process_rate:
            return

        self.camera_image = msg

        self.last_img_processed = timer()
        light_wp, state = self.process_traffic_lights()
        rospy.logwarn('INFERENCE TIME: {:.3f} s'.format(timer()-self.last_img_processed))

        '''
        Collect Training Data
        '''
        self.img_count += 1
        label = tl_utils.tl_state_to_label(state)
        if COLLECT_TD and light_wp != -1 and self.img_count % TD_RATE == 0:
            cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")
            tl_utils.save_td(uid=self.td_id, cv_image=cv_image, label=label, csv_path=OUTPUT_PATH,
                             img_path=self.td_img_path)
            self.td_id += 1

        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''

        # Ensure that the light state hasn't changed before taking any option
        if self.last_state != state:
            self.state_count = 0
            self.last_state = state

        self.state_count += 1

        if self.state_count >= self.state_count_th:
            self.last_state = state
            light_wp = light_wp if state == TrafficLight.RED else -1
            self.last_wp = light_wp

        self.upcoming_red_light_pub.publish(Int32(self.last_wp))

    def get_closest_waypoint(self, x, y):
        """Identifies the closest path waypoint to the given position
        Args:
            x: x position
            y: y position

        Returns:
            int: index of the closest waypoint in self.waypoints
        """
        return self.waypoint_tree.query([x, y], 1)[1]

    def get_light_state(self, light):
        """
        Determines the current color of the traffic light
        :param light: light to classify
        :return: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """

        # Return light state as ground truth
        if COLLECT_TD and not self.is_site:
            return light.state
        else:
            # convert camera image to cv image
            cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

            # Get classification
            return self.classifier.get_classification(cv_image, min_score=0.55)

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        closest_light = None
        line_wp_idx = None

        # List of positions that correspond to the line to stop in front of for a given intersection
        stop_line_positions = self.config['stop_line_positions']

        if self.pose:
            car_wp_idx = self.get_closest_waypoint(self.pose.pose.position.x, self.pose.pose.position.y)

            # find the closest visible traffic light (if one exists)
            diff = len(self.waypoints.waypoints)

            for i, light in enumerate(self.lights):
                # Get stop line waypoint index
                line = stop_line_positions[i]
                temp_wp_idx = self.get_closest_waypoint(line[0], line[1])
                # Find closest stop line waypoint index
                d = temp_wp_idx - car_wp_idx
                if d >= 0 and d < diff:
                    diff = d
                    closest_light = light
                    line_wp_idx = temp_wp_idx

        '''
        elif self.is_site:
            self.process_count += 1
            state = self.get_light_state(None)

            if (self.process_count % LOGGING_THROTTLE_FACTOR) == 0:
                if state is not TrafficLight.UNKNOWN:
                    rospy.logwarn("Detected {color} traffic light".format(
                        color=StateToString[state]))
        '''

        if closest_light:
            distance = line_wp_idx - car_wp_idx
            if distance <= WAYPOINT_DIFFERENCE:
                self.process_count += 1
                state = self.get_light_state(closest_light)

                if (self.process_count % LOGGING_THROTTLE_FACTOR) == 0:
                    rospy.logwarn("TRAFFIC: Detected {color} traffic light at {idx}, distance={distance} ".format(
                        idx=line_wp_idx, color=StateToString[state], distance=distance))

                return line_wp_idx, state

        return -1, TrafficLight.UNKNOWN


if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
