import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from yolo2.darknet import Darknet
from yolo3.yolov3_models import YOLOv3Darknet


def load_models(**kwargs):
    if kwargs['name'] == 'yolov2':
        darknet_model = Darknet(kwargs['cfgfile'])
        darknet_model.load_weights(kwargs['weightfile'])

    elif kwargs['name'] == 'yolov3':
        darknet_model = YOLOv3Darknet(kwargs['cfgfile'])
        darknet_model.load_darknet_weights(kwargs['weightfile'])
    else:
        raise ValueError
    return darknet_model

def extract(outputs):
    loss = 0.0
    valid_num = 0
    for opt in outputs:
        if len(opt['scores']) > 0:
            max_prob2 = opt['scores'][opt['labels'] == 1].max()
            max_prob2 = -torch.log(1.0 / max_prob2 - 1.0)
            loss = loss + max_prob2
            valid_num = valid_num + 1
    if valid_num > 0:
        loss = loss / valid_num
    return loss

class model_ensemble(nn.Module):
    def __init__(self, **kwargs):
        super(model_ensemble, self).__init__()
        self.yolov2 = load_models(**kwargs['v2kwargs']).eval()
        self.yolov3 = load_models(**kwargs['v3kwargs']).eval()
        faster_rcnn_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True).eval()
        mask_rcnn_model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True).eval()
        self.model_list = nn.ModuleList([faster_rcnn_model, mask_rcnn_model])
        self.models = kwargs['models']

    def forward(self, input):
        output_yolov2 = self.yolov2(input)
        output_yolov3 = self.yolov3(input)
        models_output = [extract(model(input)) for model in self.model_list]
        loss = 0.0
        # if 'yolov2' in self.models:
        #     loss = loss + output_yolov2.max(1)[0].mean()
        # if 'yolov3' in self.models:
        #     output_yolov3 = [m.max(1)[0] for m in output_yolov3]
        #     loss = loss + torch.mean(torch.cat(output_yolov3, 0))
        for li in models_output:
            loss = loss + li
        return output_yolov2, output_yolov3, loss





