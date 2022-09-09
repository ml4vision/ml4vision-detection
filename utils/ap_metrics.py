import brambox as bb
from brambox.io.parser import DetectionParser
from brambox.io.parser import AnnotationParser
import uuid

class APMeter:

    def __init__(self):
        self.detections = DetectionParser()
        self.annotations = AnnotationParser()

    def add_detections(self, pred_boxes, pred_scores, gt_boxes, image_id=None):
        image_id = image_id if image_id else str(uuid.uuid4())
        
        self.detections.append_image(image_id)
        for box, sc in zip(pred_boxes, pred_scores):
            box = box.tolist()
            x, y, w, h = box[0], box[1], box[2]-box[0] + 1, box[3] - box[1] + 1
            self.detections.append(image_id, 
                class_label=box[4],
                x_top_left=x,
                y_top_left=y,
                width=w,
                height=h,
                confidence=sc.item())

        self.annotations.append_image(image_id)
        for box in gt_boxes:
            x, y, w, h = box[0], box[1], box[2]-box[0] + 1, box[3] - box[1] + 1
            self.annotations.append(image_id, 
                class_label=box[4],
                x_top_left=x,
                y_top_left=y,
                width=w,
                height=h)

    def get_ap(self):
        det = self.detections.get_df()
        ann = self.annotations.get_df()
        
        # calculate ap
        # ap = bb.stat.ap(pr)
        coco = bb.eval.COCO(det, ann)
        ap = coco.mAP_coco

        # calculate best F1
        pr = bb.stat.pr(det, ann, 0.5)
        f1 = bb.stat.fscore(pr)
        threshold = bb.stat.peak(f1)

        # Find point on PR-curve that matches the computed detection threshold
        pr_point = bb.stat.point(pr, threshold.confidence)

        return ap, dict(confidence=threshold.confidence, 
                        f1=threshold.f1, 
                        precision=pr_point.precision, 
                        recall=pr_point.recall)
