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
from src.utils import tl_utils
import os
import tf
from light_classification.tl_classifier import TLClassifier

STATE_COUNT_THRESHOLD = 2
TEST_MODE_ENABLED = False
LOGGING_THROTTLE_FACTOR = 2
CAMERA_IMG_PROCESS_RATE = .8  # ms
WAYPOINT_DIFFERENCE = 300

COLLECT_TD = False
TD_RATE = 5  # only save every i-th image
TD_PATH = '/home/basti/Udacity/CarND-Capstone/sim_datasets/raw/tl-set-3'
TL_DEBUG = False


class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
        self.waypoints_2d = None
        self.waypoint_tree = None
        self.camera_image = None
        self.lights = []
        self.has_image = False

        self.td_id = 1
        self.img_count = 0

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)

        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb, queue_size=1)
        # sub6 = rospy.Subscriber('/image_color', Image, self.image_cb, queue_size=1, buff_size=2*52428800)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)
	

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()

        # TODO: Switch classifier if on site
	if self.config["is_site"] == False:
        	self.classifier = TLClassifier('/home/basti/tools/models/research/object_detection/training/fine_tuned_model/frozen_inference_graph.pb', label_map_path='/home/basti/Udacity/CarND-Capstone/sim_datasets/tl_label_map.pbtxt')
	else:	
		# TODO: Put here the site classifier        	
		self.classifier = TLClassifier('/home/basti/tools/models/research/object_detection/training/fine_tuned_model/frozen_inference_graph.pb', label_map_path='/home/basti/Udacity/CarND-Capstone/sim_datasets/tl_label_map.pbtxt')	

        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        self.class_count = 0
        self.process_count = 0
        self.last_img_processed = 0

        self.td_base_path = TD_PATH
        self.td_img_path = os.path.join(self.td_base_path, 'IMG')

        if COLLECT_TD:
            if not os.path.exists(self.td_base_path):
                os.mkdir(self.td_base_path)

            if not os.path.exists(self.td_img_path):
                os.mkdir(self.td_img_path)

        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints
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

        time_elapsed = timer() - self.last_img_processed 

        # Do not process the camera image unless 20 milliseconds have passed from last processing
        if time_elapsed < CAMERA_IMG_PROCESS_RATE:
            return

        self.has_image = True
        self.camera_image = msg

        self.last_img_processed = timer()
        light_wp, state = self.process_traffic_lights()
        print(timer() - self.last_img_processed)

        '''
        Collect Training Data
        '''
        # Collect training data
        self.img_count += 1
        label = tl_utils.tl_state_to_label(state)
        if COLLECT_TD and light_wp != -1 and self.img_count % TD_RATE == 0:
            cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")
            tl_utils.save_td(uid=self.td_id, cv_image=cv_image, label=label, csv_path=self.td_base_path,
                             img_path=self.td_img_path)
            self.td_id += 1

        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        # Ensure that the light state hasn't changed before taking any option
        if self.state != state:
            self.state_count = 0
            self.state = state
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state
            light_wp = light_wp if state == TrafficLight.RED else -1
            self.last_wp = light_wp
            self.upcoming_red_light_pub.publish(Int32(light_wp))
        else:
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
        self.state_count += 1

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

        # For test mode, just return the light state
        if TEST_MODE_ENABLED or COLLECT_TD:
            classification = light.state
        else:
            cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

            # Get classification
            classification = self.classifier.get_classification(cv_image)

        return classification

    def to_string(self, state):
        out = "unknown"
        if state == TrafficLight.GREEN:
            out = "green"
        elif state == TrafficLight.YELLOW:
            out = "yellow"
        elif state == TrafficLight.RED:
            out = "red"
        return out

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
            # car_position = self.get_closest_waypoint(self.pose.pose)
            car_wp_idx = self.get_closest_waypoint(self.pose.pose.position.x, self.pose.pose.position.y)

            # TODO find the closest visible traffic light (if one exists)
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

        if closest_light and ((line_wp_idx - car_wp_idx) <= WAYPOINT_DIFFERENCE):
            self.process_count += 1
            state = self.get_light_state(closest_light)

            if (self.process_count % LOGGING_THROTTLE_FACTOR) == 0:
                rospy.logwarn("Traffic Light: line_wp_idx={}, state={}".format(line_wp_idx, self.to_string(state)))

            return line_wp_idx, state
        else:
            # rospy.loginfo ("ptf: unknown state of traffic light")
            return -1, TrafficLight.UNKNOWN


if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
