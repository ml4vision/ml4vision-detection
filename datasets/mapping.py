import numpy as np

def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1,-n:n+1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def get_output_maps(boxes, w, h):

    heatmap = np.zeros((1, h, w), dtype=np.float32)
    scalemap = np.zeros((2, h, w), dtype=np.float32)
    classmap = -1 * np.ones((1, h, w), dtype=np.int64)

    for box in boxes:
        cx, cy = int((box[0] +  box[2])//2), int((box[1] + box[3])//2)
        bw, bh = abs(box[2] - box[0]), abs(box[3] - box[1])
        
        radius = int((min(bw, bh))//2)
        diameter = 2 * radius + 1
        gaussian = gaussian2D((diameter, diameter), diameter/6)

        left, right = min(cx, radius), min(w - cx, radius + 1)
        top, bottom = min(cy, radius), min(h - cy, radius + 1)

        masked_heatmap  = heatmap[:, cy - top:cy + bottom, cx - left:cx + right]
        masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
        
        if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
            heatmap[:, cy - top:cy + bottom, cx - left:cx + right] = np.maximum(masked_heatmap, masked_gaussian)

        # add scale and class
        if cx > 0 and cx < w and cy > 0 and cy < h:
            scalemap[0,cy,cx] = bw
            scalemap[1,cy,cx] = bh
            classmap[0,cy,cx] = box[4]

    return heatmap, scalemap, classmap

def mapping(sample):

    # combinate boxes and category ids
    boxes =  np.array(sample['boxes'])
    category_ids = np.array(sample['category_ids'])
    if len(boxes) > 0:
        boxes_combined = np.concatenate([boxes, category_ids[:,None]], axis=1)
    else:
        boxes_combined = np.array([])

    im_h, im_w = sample['image'].shape[1:]
    heatmap, scalemap, classmap = get_output_maps(boxes_combined, im_w, im_h)

    return {
        'image': sample['image'],
        'boxes': boxes_combined,
        'heatmap': heatmap,
        'scalemap': scalemap,
        'classmap': classmap
    }