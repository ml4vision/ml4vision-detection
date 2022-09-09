import torch
import torchvision
from torch.nn.functional import max_pool2d


def nms(boxes, scores, classes, threshold=0.1):
    nms_keep = torchvision.ops.nms(boxes, scores, threshold)
    boxes_keep =  torch.index_select(boxes, 0, nms_keep)
    classes_keep = torch.index_select(classes, 0, nms_keep)
    scores_keep = torch.index_select(scores, 0, nms_keep)
    return torch.cat([boxes_keep, classes_keep.unsqueeze(1)], axis=1), scores_keep

def parse_detections(pred, threshold=0.5):

    # get heatmap
    heatmap = pred[0:1, :, :]
    heatmap = torch.sigmoid(heatmap)

    # get scalemap
    scalemap = pred[1:3, :, :]

    heatmap_max = max_pool2d(heatmap, (7, 7), stride=1, padding=3)  # 1 x h x w
    keep = (heatmap_max == heatmap) * (heatmap > threshold)  # 1 x h x w

    heatmap = heatmap * keep  # 1 x h x w
    cycx = torch.nonzero(heatmap.squeeze())  # 2 x N
    scales = scalemap[keep.expand_as(scalemap)].view(2, -1)  # 2 x N

    w = scales[0, :]  # N
    h = scales[1, :]  # N

    x0 = cycx[:, 1] - w / 2  # N
    y0 = cycx[:, 0] - h / 2  # N

    x1 = cycx[:, 1] + w / 2  # N
    y1 = cycx[:, 0] + h / 2  # N

    boxes = torch.stack((x0, y0, x1, y1), dim=1)
    scores = heatmap[keep]
    if pred.size(0) > 3:
        classmap = torch.argmax(pred[3:,:,:],dim=0, keepdim=True)
        classes = classmap[keep]
    else:
        classes = torch.zeros_like(scores)

    return nms(boxes, scores, classes, threshold=0.5)


def parse_batch_detections(preds, threshold=0.5):
    detections, scores = [], []
    for pred in preds:
        det, sc = parse_detections(pred, threshold=threshold)
        detections.append(det)
        scores.append(sc)
    return detections, scores