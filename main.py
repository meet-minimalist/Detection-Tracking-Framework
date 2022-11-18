##
# @author Meet Patel <patelmeet2012@gmail.com>
# @main.py Entry point python file to detect and track objects.
# @desc Created on 2022-11-15 6:29:26 pm
# @copyright APPI SASU
##

import cv2
import config

from detector import Detector
from tracker import Tracker
from utils import get_class_mapping_dict


image_path = "../data/sample.jpg"
# vid_path = "../data/sample.mp4"
vid_path = "M:/Company/cSAT Technologies test/Task-1/tracking_video.mp4"


class_mapping_dict = get_class_mapping_dict(config.class_mapping_file)

detector_model = Detector(
    config.model_path,
    config.input_name,
    config.output_name,
    config.detection_iou_threshold,
    config.detection_score_threshold,
    ["car"],
    class_mapping_dict,
)
tracker = Tracker(iou_threshold=config.tracker_iou_threshold)

video = cv2.VideoCapture(vid_path)

ctr = 0
while True:
    ok, frame = video.read()
    if not ok:
        break

    if ctr % config.frame_interval == 0:
        res = detector_model.detect_boxes(frame)
        idx_mapping = tracker.initialize_tracker(frame, res)
    else:
        idx_mapping = tracker.get_predictions(frame)

    for single_bb_data in idx_mapping:
        pred_id, pred_bb = single_bb_data
        x1, y1, w, h = [int(c) for c in pred_bb[:4]]
        cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)

        text_string = f"id: {pred_id}"
        (test_width, text_height), baseline = cv2.getTextSize(
            text_string, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 1
        )

        cv2.putText(
            frame,
            text_string,
            (x1, y1 - baseline),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

    frame = cv2.resize(frame, (640, 480))
    cv2.imshow("img", frame)
    k = cv2.waitKey(10) & 0xFF
    if k == 27:
        break

    ctr += 1

cv2.destroyAllWindows()
