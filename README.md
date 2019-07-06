This is the project repo for the final project of the Udacity Self-Driving Car Nanodegree: Programming a Real Self-Driving Car. For more information about the project, see the project introduction [here](https://classroom.udacity.com/nanodegrees/nd013/parts/6047fe34-d93c-4f50-8336-b70ef10cb4b2/modules/e1a23b06-329a-4684-a717-ad476f0d8dff/lessons/462c933d-9f24-42d3-8bdc-a08a5fc866e4/concepts/5ab4b122-83e6-436d-850f-9f4d26627fd9).

# Simulator
Video available [here](https://youtu.be/eOcRwx-1ZvY).

# Worklow: Create your training data for the Simulator

## Record data
If `COLLECT_TD` is set to `TRUE` the `tl_detector` node will save both image data and labels. The Simulator provides labels in `light.state`.

1. Open `<path>/ros/src/tl_detector/tl_detecor.py`
2. Set `COLLECT_TD = True`
3. Set `TD_PATH` to the desired output path of the training data
4. Run the ros nodes:
```
cd <path>/ros
catkin_make
source devel/setup.sh
roslaunch launch/styx.launch
```
5. Run the Simulator and activate the camera. You can either drive manually or use the waypoint follower.

## Label your data
### Using *AutoLabel*
*AutoLabel* uses a pretrained model to detect traffic lights. For each example image a `.xml` is created containing the box locations and labels for each detected traffic light.

**CAUTION:** Since the Simulator only provides us with one label per image, each detected traffic light is automatically associated with that label. The actual color of the traffic light is not detected!

1. Run `python <path>/ros/src/tl_detector/autolabel_dataset.py --[OPTIONS]`
2. Validate the labels using *LabelImg* (see next section)

## Using *LabelImg*
LabelImg is a tool to manually add annotations such as box locations and labels to an image. LabelImg may be used to manually label the dataset or to visualize and modify the labels retrieved by the *AutoLabel* utility.

1. Run *LabelImg*
2. Select `<PATH_TO_DATASET>/voc-labels/` as annotation folder
3. Select `<PATH_TO_DATASET>/IMG` as input folder

Useful shortcuts:
* Next image: `d`
* Previous image: `a`
* Draw Box: `w`

Download *LabelImg* [here](https://github.com/tzutalin/labelImg).

## Split the Dataset into Training and Validation Data
Run `python <path>/ros/src/tl_detector/light_classification/training/train_test_split.py --[OPTIONS]`. This creates a new folder containing two datatsets `train_set` and `validation_set` 

Hint: run `python train_test_split.py -h` to list available options.

## Convert the Training and Validation Data to TFRecord
Run `python <path>/ros/src/tl_detector/light_classification/training/create_pascal_tf_record.py --[OPTIONS]`

Hint: run `python create_pascal_tf_record.py -h` to list available options.
 


 
