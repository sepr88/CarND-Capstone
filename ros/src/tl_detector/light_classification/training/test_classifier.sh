#!/usr/bin/env bash
python test_classifier.py --img_path=/home/basti/Desktop/rosbags/rosbag-4/exported --output_path=/home/basti/Desktop/rosbags/rosbag-4/result --model=/home/basti/Udacity/CarND-Capstone/data/site_frozen_inference_graph.pb --label_map=/home/basti/tensorflow/models/research/object_detection/training_site/tl_label_map.pbtxt
