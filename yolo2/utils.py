import sys
import os
import time
import math
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torch.autograd import Variable

import struct  # get_image_size
import imghdr  # get_image_size
import torchvision
import tqdm


def sigmoid(x):
    return 1.0 / (math.exp(-x) + 1.)


def softmax(x):
    x = torch.exp(x - torch.max(x))
    x = x / x.sum()
    return x


def bbox_iou(box1, box2, x1y1x2y2=True):
    if x1y1x2y2:
        mx = min(box1[0], box2[0])
        Mx = max(box1[2], box2[2])
        my = min(box1[1], box2[1])
        My = max(box1[3], box2[3])
        w1 = box1[2] - box1[0]
        h1 = box1[3] - box1[1]
        w2 = box2[2] - box2[0]
        h2 = box2[3] - box2[1]
    else:
        mx = min(box1[0] - box1[2] / 2.0, box2[0] - box2[2] / 2.0)
        Mx = max(box1[0] + box1[2] / 2.0, box2[0] + box2[2] / 2.0)
        my = min(box1[1] - box1[3] / 2.0, box2[1] - box2[3] / 2.0)
        My = max(box1[1] + box1[3] / 2.0, box2[1] + box2[3] / 2.0)
        w1 = box1[2]
        h1 = box1[3]
        w2 = box2[2]
        h2 = box2[3]
    uw = Mx - mx
    uh = My - my
    cw = w1 + w2 - uw
    ch = h1 + h2 - uh
    carea = 0
    if cw <= 0 or ch <= 0:
        return 0.0

    area1 = w1 * h1
    area2 = w2 * h2
    carea = cw * ch
    uarea = area1 + area2 - carea
    return carea / uarea


def bbox_ious(boxes1, boxes2, x1y1x2y2=True):
    if x1y1x2y2:
        mx = torch.min(boxes1[0], boxes2[0])
        Mx = torch.max(boxes1[2], boxes2[2])
        my = torch.min(boxes1[1], boxes2[1])
        My = torch.max(boxes1[3], boxes2[3])
        w1 = boxes1[2] - boxes1[0]
        h1 = boxes1[3] - boxes1[1]
        w2 = boxes2[2] - boxes2[0]
        h2 = boxes2[3] - boxes2[1]
    else:
        mx = torch.min(boxes1[0] - boxes1[2] / 2.0, boxes2[0] - boxes2[2] / 2.0)
        Mx = torch.max(boxes1[0] + boxes1[2] / 2.0, boxes2[0] + boxes2[2] / 2.0)
        my = torch.min(boxes1[1] - boxes1[3] / 2.0, boxes2[1] - boxes2[3] / 2.0)
        My = torch.max(boxes1[1] + boxes1[3] / 2.0, boxes2[1] + boxes2[3] / 2.0)
        w1 = boxes1[2]
        h1 = boxes1[3]
        w2 = boxes2[2]
        h2 = boxes2[3]
    uw = Mx - mx
    uh = My - my
    cw = w1 + w2 - uw
    ch = h1 + h2 - uh
    mask = ((cw <= 0) + (ch <= 0) > 0)
    area1 = w1 * h1
    area2 = w2 * h2
    carea = cw * ch
    carea[mask] = 0
    uarea = area1 + area2 - carea
    return carea / uarea


def bbox_iou_mat(boxes1, boxes2, x1y1x2y2=True):
    if boxes1.dim() == 1:
        boxes1 = boxes1.unsqueeze(0)
    if boxes2.dim() == 1:
        boxes2 = boxes2.unsqueeze(0)
    boxes1 = boxes1.unsqueeze(1).expand(-1, boxes2.shape[0], -1)
    boxes2 = boxes2.unsqueeze(0).expand(boxes1.shape[0], -1, -1)

    if x1y1x2y2:
        mx = torch.minimum(boxes1[..., 0], boxes2[..., 0])
        Mx = torch.maximum(boxes1[..., 2], boxes2[..., 2])
        my = torch.minimum(boxes1[..., 1], boxes2[..., 1])
        My = torch.maximum(boxes1[..., 3], boxes2[..., 3])
        w1 = boxes1[..., 2] - boxes1[..., 0]
        h1 = boxes1[..., 3] - boxes1[..., 1]
        w2 = boxes2[..., 2] - boxes2[..., 0]
        h2 = boxes2[..., 3] - boxes2[..., 1]
    else:
        mx = torch.minimum(boxes1[..., 0] - boxes1[..., 2] / 2.0, boxes2[..., 0] - boxes2[..., 2] / 2.0)
        Mx = torch.maximum(boxes1[..., 0] + boxes1[..., 2] / 2.0, boxes2[..., 0] + boxes2[..., 2] / 2.0)
        my = torch.minimum(boxes1[..., 1] - boxes1[..., 3] / 2.0, boxes2[..., 1] - boxes2[..., 3] / 2.0)
        My = torch.maximum(boxes1[..., 1] + boxes1[..., 3] / 2.0, boxes2[..., 1] + boxes2[..., 3] / 2.0)
        w1 = boxes1[..., 2]
        h1 = boxes1[..., 3]
        w2 = boxes2[..., 2]
        h2 = boxes2[..., 3]
    uw = Mx - mx
    uh = My - my
    cw = w1 + w2 - uw
    ch = h1 + h2 - uh
    mask = ((cw <= 0) + (ch <= 0) > 0)
    area1 = w1 * h1
    area2 = w2 * h2
    carea = cw * ch
    carea[mask] = 0
    uarea = area1 + area2 - carea
    return carea / uarea

# def nms(boxes, nms_thresh):
#     if len(boxes) == 0:
#         return boxes
#
#     det_confs = torch.zeros(len(boxes))
#     for i in range(len(boxes)):
#         det_confs[i] = 1-boxes[i][4]
#
#     _, sortIds = torch.sort(det_confs)
#     out_boxes = []
#     for i in range(len(boxes)):
#         box_i = boxes[sortIds[i]]
#         if box_i[4] > 0:
#             out_boxes.append(box_i)
#             for j in range(i+1, len(boxes)):
#                 box_j = boxes[sortIds[j]]
#                 if bbox_iou(box_i, box_j, x1y1x2y2=False) > nms_thresh:
#                     #print(box_i, box_j, bbox_iou(box_i, box_j, x1y1x2y2=False))
#                     box_j[4] = 0
#     return out_boxes

# Improved by W
def nms(boxes, nms_thresh):
    if len(boxes) == 0:
        return boxes

    _, sortIds = torch.sort(boxes[:, 4], descending=True)
    boxes = boxes[sortIds]
    boxes = boxes.cpu()

    out_boxes = []
    for i in range(len(boxes)):
        box_i = boxes[i]
        if box_i[4] > 0:
            out_boxes.append(box_i)
            box_j = boxes[i + 1:]
            ids = bbox_ious(box_i, box_j.t(), x1y1x2y2=False) > nms_thresh
            box_j[ids, 4] = 0
    out_boxes = torch.stack(out_boxes, 0)
    return out_boxes


def lab2box(x):
    y = x.new(size=(x.shape[0], 7))
    y[:, :4] = x[:, 1:5]
    y[:, -1] = x[:, 0]
    y[:, 4:6] = 1.0
    return y


def ap_per_class(tp, conf, pred_cls, target_cls):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (list).
        conf:  Objectness value from 0-1 (list).
        pred_cls: Predicted object classes (list).
        target_cls: True object classes (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)

    # Create Precision-Recall curve and compute AP for each class
    ap, p, r = [], [], []
    # for c in tqdm.tqdm(unique_classes, desc="Computing AP"):
    c = unique_classes[0]  # class person
    i = pred_cls == c
    n_gt = (target_cls == c).sum()  # Number of ground truth objects
    n_p = i.sum()  # Number of predicted objects

    if n_p == 0 and n_gt == 0:
        ap, p, r = [], [], []
    elif n_p == 0 or n_gt == 0:
        ap.append(0)
        r.append(0)
        p.append(0)
    else:
        # Accumulate FPs and TPs
        fpc = (1 - tp[i]).cumsum()
        tpc = (tp[i]).cumsum()

        # Recall
        recall_curve = tpc / (n_gt + 1e-16)
        r.append(recall_curve[-1])

        # Precision
        precision_curve = tpc / (tpc + fpc)
        p.append(precision_curve[-1])

        # AP from recall-precision curve
        ap.append(compute_ap(recall_curve, precision_curve))

    # Compute F1 score (harmonic mean of precision and recall)
    p, r, ap = np.array(p), np.array(r), np.array(ap)
    f1 = 2 * p * r / (p + r + 1e-16)

    # return p, r, ap, f1, unique_classes.astype("int32")
    return precision_curve, recall_curve


# above

def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def get_batch_statistics(outputs, targets, iou_threshold):
    """ Compute true positives, predicted scores and predicted labels per sample """
    batch_metrics = []
    for sample_i in range(len(outputs)):

        if outputs[sample_i] is None:
            continue

        output = outputs[sample_i]
        pred_boxes = output[:, :4]
        pred_scores = output[:, 4]
        pred_labels = output[:, -1]

        true_positives = np.zeros(pred_boxes.shape[0])

        annotations = targets[targets[:, 0] == sample_i][:, 1:]
        target_labels = annotations[:, 0] if len(annotations) else []
        if len(annotations):
            detected_boxes = []
            target_boxes = annotations[:, 1:]

            for pred_i, (pred_box, pred_label) in enumerate(zip(pred_boxes, pred_labels)):

                # If targets are found break
                if len(detected_boxes) == len(annotations):
                    break

                # Ignore if label is not one of the target labels
                if pred_label not in target_labels:
                    continue

                iou, box_index = YOLOv3bbox_iou(pred_box.unsqueeze(0), target_boxes).max(0)
                if iou >= iou_threshold and box_index not in detected_boxes:
                    true_positives[pred_i] = 1
                    detected_boxes += [box_index]
        batch_metrics.append([true_positives, pred_scores.cpu().numpy(), pred_labels.cpu().numpy()])
    return batch_metrics


# def YOLOv3bbox_iou(box1, box2, x1y1x2y2=True):
#     """
#     Returns the IoU of two bounding boxes
#     """
#     if not x1y1x2y2:
#         # Transform from center and width to exact coordinates
#         b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
#         b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
#         b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
#         b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
#     else:
#         # Get the coordinates of bounding boxes
#         b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
#         b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]
#
#     # get the corrdinates of the intersection rectangle
#     inter_rect_x1 = torch.max(b1_x1, b2_x1)
#     inter_rect_y1 = torch.max(b1_y1, b2_y1)
#     inter_rect_x2 = torch.min(b1_x2, b2_x2)
#     inter_rect_y2 = torch.min(b1_y2, b2_y2)
#     # Intersection area
#     inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
#         inter_rect_y2 - inter_rect_y1 + 1, min=0
#     )
#     # Union Area
#     b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
#     b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)
#
#     iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)
#
#     return iou

def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, labels=()):
    """Performs Non-Maximum Suppression (NMS) on inference results
    Returns:
         detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
    """

    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_det = 300  # maximum number of detections per image
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 1.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 5), device=x.device)
            v[:, :4] = l[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded

    return output


def convert2cpu(gpu_matrix):
    return torch.FloatTensor(gpu_matrix.size()).copy_(gpu_matrix)


def convert2cpu_long(gpu_matrix):
    return torch.LongTensor(gpu_matrix.size()).copy_(gpu_matrix)


def get_region_boxes_general(output, model, conf_thresh, name='yolov2', img_size=416, lab_filter=None):
    if name == 'yolov2':
        num_classes = model.num_classes
        anchors = model.anchors
        num_anchors = model.num_anchors
        if isinstance(output, list):
            assert len(output) == 1
            output = output[0]
        all_boxes = get_region_boxes(output, conf_thresh, num_classes, anchors, num_anchors, name=name)
    else:
        raise ValueError
    if lab_filter is not None:
        for i in range(len(all_boxes)):
            all_boxes[i] = all_boxes[i][all_boxes[i][:, 6] == lab_filter]
    return all_boxes


def get_region_boxes(output, conf_thresh, num_classes, anchors, num_anchors, only_objectness=1, validation=False,
                     name=None):
    anchor_step = len(anchors) // num_anchors
    device = output.device
    if output.dim() == 3:
        output = output.unsqueeze(0)
    batch = output.size(0)
    assert (output.size(1) == (5 + num_classes) * num_anchors)
    h = output.size(2)
    w = output.size(3)

    output = output.view(batch * num_anchors, 5 + num_classes, h * w)
    output = output.transpose(0, 1).contiguous()
    output = output.view(5 + num_classes, batch * num_anchors * h * w)
    # grid_x = torch.linspace(0, w-1, w).repeat(h,1).repeat(batch*num_anchors, 1, 1).view(batch*num_anchors*h*w).to(output)
    # grid_y = torch.linspace(0, h-1, h).repeat(w,1).t().repeat(batch*num_anchors, 1, 1).view(batch*num_anchors*h*w).to(output)
    grid_y, grid_x = torch.meshgrid([torch.arange(w, device=device), torch.arange(h, device=device)])
    grid_x = grid_x.repeat(batch * num_anchors, 1, 1).flatten()
    grid_y = grid_y.repeat(batch * num_anchors, 1, 1).flatten()
    xs = torch.sigmoid(output[0]) + grid_x
    ys = torch.sigmoid(output[1]) + grid_y

    anchor_tensor = torch.tensor(anchors, device=device).view(num_anchors, anchor_step)
    # anchor_w = anchor_tensor.index_select(1, torch.LongTensor([0]))
    # anchor_h = anchor_tensor.index_select(1, torch.LongTensor([1]))
    anchor_w = anchor_tensor[:, 0:1]
    anchor_h = anchor_tensor[:, 1:2]
    anchor_w = anchor_w.repeat(batch, 1).repeat(1, 1, h * w).view(batch * num_anchors * h * w)
    anchor_h = anchor_h.repeat(batch, 1).repeat(1, 1, h * w).view(batch * num_anchors * h * w)
    ws = torch.exp(output[2]) * anchor_w
    hs = torch.exp(output[3]) * anchor_h

    det_confs = torch.sigmoid(output[4])
    # cls_confs = torch.nn.Softmax()(Variable(output[5:5+num_classes].transpose(0,1))).data

    if name == 'yolov2':
        cls_confs = output[5:5 + num_classes].transpose(0, 1).softmax(-1)
    else:
        raise ValueError

    cls_max_confs, cls_max_ids = torch.max(cls_confs, 1)
    cls_max_confs = cls_max_confs.view(-1)
    cls_max_ids = cls_max_ids.view(-1)

    raw_boxes = torch.stack([xs/w, ys/h, ws/w, hs/h, det_confs, cls_max_confs, cls_max_ids], 1).view(batch, -1, 7)
    if only_objectness:
        conf = det_confs
    else:
        conf = det_confs * cls_max_confs
    inds = (conf > conf_thresh).view(batch, -1)

    all_boxes = [b[i] for b, i in zip(raw_boxes, inds)]

    if (not only_objectness) and validation:
        raise NotImplementedError

    # sz_hw = h * w
    # sz_hwa = sz_hw * num_anchors
    # det_confs = convert2cpu(det_confs)
    # cls_max_confs = convert2cpu(cls_max_confs)
    # cls_max_ids = convert2cpu_long(cls_max_ids)
    # xs = convert2cpu(xs)
    # ys = convert2cpu(ys)
    # ws = convert2cpu(ws)
    # hs = convert2cpu(hs)
    # if validation:
    #     cls_confs = convert2cpu(cls_confs.view(-1, num_classes))
    #
    # all_boxes = []
    # for b in range(batch):
    #     boxes = []
    #     for cy in range(h):
    #         for cx in range(w):
    #             for i in range(num_anchors):
    #                 ind = b * sz_hwa + i * sz_hw + cy * w + cx
    #                 det_conf = det_confs[ind]
    #                 if only_objectness:
    #                     conf = det_confs[ind]
    #                 else:
    #                     conf = det_confs[ind] * cls_max_confs[ind]
    #
    #                 if conf > conf_thresh:
    #                     bcx = xs[ind]
    #                     bcy = ys[ind]
    #                     bw = ws[ind]
    #                     bh = hs[ind]
    #                     cls_max_conf = cls_max_confs[ind]
    #                     cls_max_id = cls_max_ids[ind]
    #                     box = [bcx / w, bcy / h, bw / w, bh / h, det_conf, cls_max_conf, cls_max_id]
    #                     if (not only_objectness) and validation:
    #                         for c in range(num_classes):
    #                             tmp_conf = cls_confs[ind][c]
    #                             if c != cls_max_id and det_confs[ind] * tmp_conf > conf_thresh:
    #                                 box.append(tmp_conf)
    #                                 box.append(c)
    #                     boxes.append(box)
    #     all_boxes.append(boxes)
    # all_boxes = [torch.tensor(ab) for ab in all_boxes]

    return all_boxes


def plot_boxes_cv2(img, boxes, savename=None, class_names=None, color=None):
    import cv2
    colors = torch.FloatTensor([[1, 0, 1], [0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 1, 0], [1, 0, 0]]);

    def get_color(c, x, max_val):
        ratio = float(x) / max_val * 5
        i = int(math.floor(ratio))
        j = int(math.ceil(ratio))
        ratio = ratio - i
        r = (1 - ratio) * colors[i][c] + ratio * colors[j][c]
        return int(r * 255)

    width = img.shape[1]
    height = img.shape[0]
    for i in range(len(boxes)):
        box = boxes[i]
        x1 = int(round((box[0] - box[2] / 2.0) * width))
        y1 = int(round((box[1] - box[3] / 2.0) * height))
        x2 = int(round((box[0] + box[2] / 2.0) * width))
        y2 = int(round((box[1] + box[3] / 2.0) * height))

        if color:
            rgb = color
        else:
            rgb = (255, 0, 0)
        if len(box) >= 7 and class_names:
            cls_conf = box[5]
            cls_id = box[6]
            print('%s: %f' % (class_names[cls_id], cls_conf))
            classes = len(class_names)
            offset = cls_id * 123457 % classes
            red = get_color(2, offset, classes)
            green = get_color(1, offset, classes)
            blue = get_color(0, offset, classes)
            if color is None:
                rgb = (red, green, blue)
            img = cv2.putText(img, class_names[cls_id], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1.2, rgb, 1)
        img = cv2.rectangle(img, (x1, y1), (x2, y2), rgb, 1)
    if savename:
        print("save plot results to %s" % savename)
        cv2.imwrite(savename, img)
    return img

def back_text(draw, x, y, msg, backc, fontc, font=None):
    if font is None:
        font = draw.getfont()
        # print(font.getsize())
    w, h = font.getsize(msg)
    draw.rectangle((x, y, x+w, y+h), fill=backc)
    draw.text((x, y), msg, fill=fontc)
    return None


def plot_boxes(img, boxes, savename=None, class_names=None, class_range=None, text='conf'):
    colors = torch.FloatTensor([[1, 0, 1], [0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 1, 0], [1, 0, 0]])
    fontc = (255, 255, 255)
    if class_range is None:
        class_range = class_names

    def get_color(c, x, max_val):
        ratio = float(x) / max_val * 5
        i = int(math.floor(ratio))
        j = int(math.ceil(ratio))
        ratio = ratio - i
        r = (1 - ratio) * colors[i][c] + ratio * colors[j][c]
        return int(r * 255)

    width = img.width
    height = img.height
    draw = ImageDraw.Draw(img)
    for i in range(len(boxes)):
        box = boxes[i]
        x1 = (box[0] - box[2] / 2.0) * width
        y1 = (box[1] - box[3] / 2.0) * height
        x2 = (box[0] + box[2] / 2.0) * width
        y2 = (box[1] + box[3] / 2.0) * height

        rgb = (255, 0, 0)
        cls_id = int(box[6])
        if class_names[cls_id] in class_range:
            if len(box) >= 7 and class_names:
                cls_conf = box[5]
                # print('[%i]%s: %f, obj conf %f' % (cls_id, class_names[cls_id], cls_conf, box[4]))
                classes = len(class_names)
                offset = cls_id * 123457 % classes
                red = get_color(2, offset, classes)
                green = get_color(1, offset, classes)
                blue = get_color(0, offset, classes)
                rgb = (red, green, blue)
                if text == 'class&conf':
                    # draw.text((x1, y1), "%s obj%.3f cls%.3f" % (class_names[cls_id], box[4], cls_conf), fill=rgb)
                    # draw.text((x1, y1), "%s %.3f" % (class_names[cls_id], box[4]), fill=rgb)
                    back_text(draw, x1, y1, "%s %.3f" % (class_names[cls_id], box[4]), backc=rgb, fontc=fontc)

                elif text == 'conf':
                    # draw.text((x1, y1), "%s obj%.3f cls%.3f" % (class_names[cls_id], box[4], cls_conf), fill=rgb)
                    # draw.text((x1, y1), "%.3f" % (box[4]), fill=rgb)
                    back_text(draw, x1, y1, "%.3f" % (box[4]), backc=rgb, fontc=fontc)
                elif text == 'class':
                    w = 12
                    h = 40
                    draw.rectangle((x1, y1 - w, x1 + h, y1), fill=rgb)
                    draw.text((x1 + 2, y1 - w), class_names[cls_id], fill=(0, 0, 0))
                elif isinstance(text, int):
                    if cls_id == text:
                        # draw.text((x1, y1), "%s %.3f" % (class_names[cls_id], box[4]), fill=rgb)
                        back_text(draw, x1, y1, "%s %.3f" % (class_names[cls_id], box[4]), backc=rgb, fontc=fontc)
                elif text is not None:
                    pass
            draw.rectangle([x1, y1, x2, y2], outline=rgb)
    if savename:
        print("save plot results to %s" % savename)
        img.save(savename)
    return img


def read_truths(lab_path):
    if not os.path.exists(lab_path):
        return np.array([])
    if os.path.getsize(lab_path):
        truths = np.loadtxt(lab_path)
        truths = truths.reshape(truths.size // 5, 5)  # to avoid single truth problem
        return truths
    else:
        return np.array([])


def read_truths_args(lab_path, min_box_scale):
    truths = read_truths(lab_path)
    new_truths = []
    # remove truths of which the width is smaller then the min_box_scale
    for i in range(truths.shape[0]):
        if truths[i][3] < min_box_scale:
            continue
        new_truths.append([truths[i][0], truths[i][1], truths[i][2], truths[i][3], truths[i][4]])
    return np.array(new_truths)


def load_class_names(namesfile):
    class_names = []
    with open(namesfile, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.rstrip()
        class_names.append(line)
    return class_names


def image2torch(img):
    width = img.width
    height = img.height
    img = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
    img = img.view(height, width, 3).transpose(0, 1).transpose(0, 2).contiguous()
    img = img.view(1, 3, height, width)
    img = img.float().div(255.0)
    return img


def do_detect(model, img, conf_thresh, nms_thresh, device='cpu', name=None, lab_filter=None, before_nms_filter=[]):
    # WhoTH changed the source code to return boxes for batch imgs

    model.eval()
    if isinstance(img, Image.Image):
        width = img.width
        height = img.height
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
        img = img.view(height, width, 3).transpose(0, 1).transpose(0, 2).contiguous()
        img = img.view(1, 3, height, width)
        img = img.float().div(255.0)
    elif type(img) == np.ndarray:  # cv2 image
        img = torch.from_numpy(img.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)
    elif type(img) == torch.Tensor:
        pass
    else:
        print("unknown image type")
        exit(-1)

    img = img.to(device)
    img = torch.autograd.Variable(img)

    output = model(img)
    # output = output.data
    # for j in range(100):
    #    sys.stdout.write('%f ' % (output.storage()[j]))
    # print('')

    # all_boxes_raw = get_region_boxes(output, conf_thresh, model.num_classes, model.anchors, model.num_anchors)
    all_boxes_raw = get_region_boxes_general(output, model, conf_thresh, name=name, lab_filter=lab_filter)
    all_boxes = []
    for boxes in all_boxes_raw:
        if len(before_nms_filter) > 0:
            ids = boxes[:, -1] == before_nms_filter[0]
            for il in before_nms_filter[1:]:
                ids = ids.logical_and(boxes[:, -1] == il)
            boxes = boxes[ids]
        all_boxes.append(nms(boxes, nms_thresh))

    return all_boxes


def read_data_cfg(datacfg):
    options = dict()
    options['gpus'] = '0,1,2,3'
    options['num_workers'] = '10'
    with open(datacfg, 'r') as fp:
        lines = fp.readlines()

    for line in lines:
        line = line.strip()
        if line == '':
            continue
        key, value = line.split('=')
        key = key.strip()
        value = value.strip()
        options[key] = value
    return options


def scale_bboxes(bboxes, width, height):
    import copy
    dets = copy.deepcopy(bboxes)
    for i in range(len(dets)):
        dets[i][0] = dets[i][0] * width
        dets[i][1] = dets[i][1] * height
        dets[i][2] = dets[i][2] * width
        dets[i][3] = dets[i][3] * height
    return dets


def file_lines(thefilepath):
    count = 0
    thefile = open(thefilepath, 'rb')
    while True:
        buffer = thefile.read(8192 * 1024)
        if not buffer:
            break
        count += buffer.count('\n')
    thefile.close()
    return count


def get_image_size(fname):
    '''Determine the image type of fhandle and return its size.
    from draco'''
    with open(fname, 'rb') as fhandle:
        head = fhandle.read(24)
        if len(head) != 24:
            return
        if imghdr.what(fname) == 'png':
            check = struct.unpack('>i', head[4:8])[0]
            if check != 0x0d0a1a0a:
                return
            width, height = struct.unpack('>ii', head[16:24])
        elif imghdr.what(fname) == 'gif':
            width, height = struct.unpack('<HH', head[6:10])
        elif imghdr.what(fname) == 'jpeg' or imghdr.what(fname) == 'jpg':
            try:
                fhandle.seek(0)  # Read 0xff next
                size = 2
                ftype = 0
                while not 0xc0 <= ftype <= 0xcf:
                    fhandle.seek(size, 1)
                    byte = fhandle.read(1)
                    while ord(byte) == 0xff:
                        byte = fhandle.read(1)
                    ftype = ord(byte)
                    size = struct.unpack('>H', fhandle.read(2))[0] - 2
                    # We are at a SOFn block
                fhandle.seek(1, 1)  # Skip `precision' byte.
                height, width = struct.unpack('>HH', fhandle.read(4))
            except Exception:  # IGNORE:W0703
                return
        else:
            return
        return width, height


def logging(message):
    print('%s %s' % (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), message))
