#!/usr/bin/env bash
raw_datasets=/home/basti/Desktop/rosbags
merged_dataset=/home/basti/Desktop/joined_rosbags
preprocessed_dataset=/home/basti/Desktop/preprocessed_rosbags
augmented_dataset=/home/basti/Desktop/augmented_rosbags
split_dataset=/home/basti/Desktop/split_rosbags
temp=/home/basti/Desktop/temp-$(date +'%Y-%m-%d-%H-%M-%S')
temp_train=${temp}/train_set
temp_validate=${temp}/validation_set
label_map_path=/home/basti/Udacity/CarND-Capstone/data/tl_label_map.pbtxt

echo Dataset generation pipeline started

# remove previous
rm -R ${merged_dataset}
mkdir ${merged_dataset}

rm -R ${split_dataset}
mkdir ${split_dataset}

# Merge datasets
echo Merging datasets in ${raw_datasets} ...
python join_datasets.py --input_path=${raw_datasets} --output_path=${merged_dataset} --img_dir=exported --label_dir=labels
echo Merging complete. Datasets saved to ${merged_dataset}

# Split dataset into train, validation and test set
echo Splitting dataset into train, validation, and test sets ...
mkdir ${temp}
mkdir ${temp_train}
mkdir ${temp_train}/img
mkdir ${temp_train}/labels
mkdir ${temp_validate}
mkdir ${temp_validate}/img
mkdir ${temp_validate}/labels

python train_test_split.py --input_path=$merged_dataset --output_path=$temp --img_dir=exported --label_dir=labels --validation_size=0.1
python train_test_split.py --input_path=$temp_train --output_path=$split_dataset --img_dir=img --label_dir=labels --validation_size=0.2
python copy_dataset.py --src=$temp_validate --dst=$split_dataset/test_set
rm -R $temp
echo Splitting dataset complete.

# Preprocess dataset (normalize, crop, convert to grayscale)
echo Preprocessing datasets ...
python preprocessor.py --label_path=$split_dataset/train_set/labels --output_path=$split_dataset/train_set/preprocessed

python preprocessor.py --label_path=$split_dataset/validation_set/labels --output_path=$split_dataset/validation_set/preprocessed
echo Preprocessing complete.

# Augment data (horizontal flip, add noise, smoothen, sharpen, random scale)
echo Augmenting dataset ...
python data_augmentation.py --label_path=${split_dataset}/train_set/preprocessed/labels --output_path=${split_dataset}/train_set/augmented

echo Data augmentation complete

# Convert datasets to tf record format
echo Generating tf records ...

python create_pascal_tf_record.py --data_dir=$split_dataset/train_set/augmented --output_path=$split_dataset/train.record --label_map_path=$label_map_path --ignore_difficult_instances=False
python create_pascal_tf_record.py --data_dir=$split_dataset/validation_set/preprocessed --output_path=$split_dataset/validation.record --label_map_path=$label_map_path --ignore_difficult_instances=False

echo Generating tf record files complete. tf records saved to $split_dataset
echo Dataset generation complete.
