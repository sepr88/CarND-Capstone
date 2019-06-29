import csv
import cv2
import os
from os.path import join
from styx_msgs.msg import TrafficLight


def save_td(uid, cv_image, label, csv_path, img_path):
    '''
    uid:        unique identifier, used as image name
    cv_image:   the image to be saved
    label:      the associated label, e.g. light.state
    path:       folder to store the img data
    '''

    img_path_full = join(img_path, 'image{uid}.png'.format(uid=uid))

    # save image
    cv2.imwrite(img_path_full, cv_image)

    # save labels (append label to csv
    with open(join(csv_path, 'labels.csv'), 'a') as f:
        f.write('{img_path_full};{label}\n'.format(img_path_full=img_path_full, label=label))
                        
    return True


def tl_state_to_label(state):
    if state == TrafficLight.RED:
        return 'red'
    
    if state == TrafficLight.GREEN:
        return 'green'
    
    if state == TrafficLight.YELLOW:
        return 'yellow'
    
    return 'off'
