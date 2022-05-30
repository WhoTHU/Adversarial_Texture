import torch.nn as nn
from yolo2.darknet import Darknet

def load_models(**kwargs):
    if kwargs['name'] == 'yolov2':
        darknet_model = Darknet(kwargs['cfgfile'])
        darknet_model.load_weights(kwargs['weightfile'])
    return darknet_model