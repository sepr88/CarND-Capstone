#!/usr/bin/env bash

# Modify
train_dir=/home/basti/tensorflow/models/research/object_detection/training_site
test_img_dir=/home/basti/Desktop/split_rosbags/test_set/img
out_img_dir=/home/basti/Desktop/split_rosbags/test_set/result
tf_record_dir=/home/basti/Desktop/split_rosbags

# Dont Modify
fine_tuned_model_dir=${train_dir}/fine_tuned_model
checkpoint_dir=${train_dir}/models/train
data_dir=${train_dir}/data
train_tf_record=${tf_record_dir}/train.record
validation_tf_record=${tf_record_dir}/validation.record

# Remove old model
if [ -d "${fine_tuned_model_dir}" ]; then
  rm -R ${fine_tuned_model_dir}
fi

# Remove old checkpoints
if [ -d "${checkpoint_dir}" ]; then
  rm -R ${checkpoint_dir}
fi

# Remove old datasets
if [ -d "${data_dir}" ]; then
  rm -R ${data_dir}
fi

mkdir ${data_dir}

# Copy train and validation sets
cp ${train_tf_record} ${data_dir}
cp ${validation_tf_record} ${data_dir}

# Source
# export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

# Train
python ${train_dir}/train.py --logtostderr --train_dir=${checkpoint_dir} --pipeline_config_path=${train_dir}/pipeline.config

# Export inference graph
python ${train_dir}/export_inference_graph.py --input_type=image_tensor --pipeline_config_path=${train_dir}/pipeline.config --trained_checkpoint_prefix=${checkpoint_dir}/model.ckpt-5000 --output_directory=${fine_tuned_model_dir}

# Test classifier
python test_classifier.py --model=${fine_tuned_model_dir}/frozen_inference_graph.pb --img_path=${test_img_dir} --out=${out_img_dir}
