def load_models(**kwargs):
    if kwargs['name'] == 'yolov2':
        from yolo2.darknet import Darknet
        darknet_model = Darknet(kwargs['cfgfile'])
        darknet_model.load_weights(kwargs['weightfile'])

    elif kwargs['name'] == 'yolov3':
        from yolo3.yolov3_models import YOLOv3Darknet
        darknet_model = YOLOv3Darknet(kwargs['cfgfile'])
        darknet_model.load_darknet_weights(kwargs['weightfile'])
    elif kwargs['name'] == 'faster_rcnn':
        import torchvision
        darknet_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True).eval()
    elif kwargs['name'] == 'mask_rcnn':
        import torchvision
        darknet_model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True).eval()
    elif kwargs['name'] == 'ensemble':
        from models_ensemble import model_ensemble
        darknet_model = model_ensemble(**kwargs)
    # elif 'mmdet' in kwargs['name']:
    #     from mmdet.apis import inference_detector, init_detector
    #     import mmcv
    #     import torch.nn as nn
    #     import numpy as np
    #
    #     class mmdet_model(nn.Module):
    #         def __init__(self,model):
    #             super(mmdet_model, self).__init__()
    #             self.model = model
    #             self.img_size = 416
    #         def forward(self, inputs):
    #             inputs = inputs.cpu().numpy().transpose(0, 2, 3, 1)
    #             inputs = (inputs * 255).astype(np.uint8)
    #             inputs = np.split(inputs, inputs.shape[0])
    #             inputs = [inp.squeeze(0) for inp in inputs]
    #             result = inference_detector(self.model, inputs)
    #             return result
    #
    #
    #     config = kwargs['config']
    #     checkpoint = kwargs['checkpoint']
    #     darknet_model = init_detector(config, checkpoint, device=kwargs['device'])
    #     darknet_model = darknet_model.eval()
    #     return mmdet_model(darknet_model)
    else:
        raise ValueError
    return darknet_model