# -*- coding: utf-8 -*-

import cv2
from os.path import join
from styx_msgs.msg import TrafficLight
import sys


def save_td(uid, cv_image, label, csv_path, img_path):
    """
    Save image and label
    @params:
        uid         - Required : unique identifier, used as image name (Var)
        cv_image:   - Required : the image to be saved (Image)
        label:      - Required : the associated label, e.g. light.state (Str)
        path:       - Required : folder to store the img data (Str)
    """

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


# Print iterations progress
def print_progress(iteration, total, prefix='', suffix='', decimals=1, bar_length=100):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        bar_length  - Optional  : character length of bar (Int)
    """
    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = '█' * filled_length + '-' * (bar_length - filled_length)

    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),

    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()

def yes_or_no(question):
    reply = str(raw_input(question + ' (y/n): ')).lower().strip()
    if reply[0] == 'y':
        return True
    if reply[0] == 'n':
        return False
    else:
        return yes_or_no(question)
