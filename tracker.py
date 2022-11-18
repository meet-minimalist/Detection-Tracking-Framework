##
# @author Meet Patel <patelmeet2012@gmail.com>
# @tracker.py Class to track the detections
# @desc Created on 2022-11-15 7:28:14 pm
# @copyright APPI SASU
##

from typing import List, Tuple, Union

import cv2
import numpy as np


# Note: tracker should be populated with x1, y1, w, h coordinates
class Tracker:
    """
    Tracker class to track the bboxes.
    """

    def __init__(self, tracker_type: str = "KCF", iou_threshold: float = 0.8):
        """Constructor of the Tracker class.

        Args:
            tracker_type (str, optional): Type of tracker to be used.
                Defaults to "KCF".
            iou_threshold (float, optional): IOU Threshold for assigning
                existing tracking ids. Defaults to 0.8.
        """
        self.tracker_type = tracker_type
        self.tracker_initialized = False
        self.unique_object_counter = 1
        self.all_trackers = []
        self.iou_threshold = iou_threshold

    def get_new_tracker(self) -> cv2.TrackerKCF:
        """Function to get the new instance of the tracker.

        Returns:
            cv2.TrackerKCF: Instance of KCF tracker.
        """
        return cv2.TrackerKCF_create()

    def iou(self, bbox_1: Union[Tuple, List], bbox_2: Union[Tuple, List]) -> float:
        """Function to get the IOU based on provided 2 bounding boxes.

        Args:
            bbox_1 (Union[Tuple, List]): Bounding box in format [x1, y1, w, h].
            bbox_2 (Union[Tuple, List]): Bounding box in format [x1, y1, w, h].

        Returns:
            float: Computed IOU value.
        """

        x1 = max(bbox_1[0], bbox_2[0])
        y1 = max(bbox_1[1], bbox_2[1])

        x2 = min(bbox_1[0] + bbox_1[2], bbox_2[0] + bbox_2[2])
        y2 = min(bbox_1[1] + bbox_1[3], bbox_2[1] + bbox_2[3])

        intersection = max(y2 - y1 + 1, 0) * max(x2 - x1 + 1, 0)

        union = (bbox_1[2] * bbox_1[3]) + (bbox_2[2] * bbox_2[3]) - intersection

        iou = intersection / (union + 1e-6)
        return iou

    def xcycwh_x1y1wh(self, box: np.ndarray) -> np.ndarray:
        """Function to convert a box from [xc, yc, w, h] format into
        [x1, y1, w, h] format.

        Args:
            box (np.ndarray): Bounding box in [xc, yc, w, h] format.

        Returns:
            np.ndarray: Updated bounding box in [x1, y1, w, h] format.
        """
        xc, yc, w, h = box[:4]
        x1 = int(xc - w / 2)
        y1 = int(yc - h / 2)
        bb_xywh = [x1, y1, int(w), int(h), box[4], box[5]]
        return bb_xywh

    def initialize_tracker(self, image: np.ndarray, boxes: np.ndarray) -> List:
        """Function to initialize new tracker based detection boxes.

        Args:
            image (np.ndarray): Image array in the shape [H, W, 3].
            boxes (np.ndarray): Detected boxes in the shape [N, 6], in the
                format of [xc, yc, w, h, class_id, class_probability].

        Returns:
            List: List with each element having tracking id and detected box in
                the shape [N, 6], in the format [x1, y1, w, h, class_id, class_probability].
        """
        if len(boxes) == 0:
            self.all_trackers = []

        idx_mapping = []
        if not self.tracker_initialized:
            self.all_trackers = []
            for bb in boxes:
                bb_xywh = self.xcycwh_x1y1wh(bb)
                # Initialize tracker with first frame and bounding box
                tracker = self.get_new_tracker()
                status = tracker.init(image, bb_xywh[:4])
                self.all_trackers.append([self.unique_object_counter, tracker])
                idx_mapping.append([self.unique_object_counter, bb_xywh])
                self.unique_object_counter += 1

            self.tracker_initialized = True
        else:
            tracker_predictions = self.get_predictions(image)
            # List of [id, [4]]

            self.all_trackers = []
            for bb in boxes:
                bb_xywh = self.xcycwh_x1y1wh(bb)

                existing_tracker_found = False
                for tracker_pred in tracker_predictions:
                    track_id, track_bbx = tracker_pred

                    iou = self.iou(track_bbx, bb_xywh)
                    if iou > self.iou_threshold:
                        new_tracker = self.get_new_tracker()
                        new_tracker.init(image, bb_xywh[:4])
                        self.all_trackers.append([track_id, new_tracker])
                        idx_mapping.append([track_id, bb_xywh])
                        existing_tracker_found = True
                        break
                    else:
                        continue

                if not existing_tracker_found:
                    new_tracker = self.get_new_tracker()
                    new_tracker.init(image, bb_xywh[:4])
                    self.all_trackers.append([self.unique_object_counter, new_tracker])
                    idx_mapping.append([self.unique_object_counter, bb_xywh])
                    self.unique_object_counter += 1

        return idx_mapping

    def get_predictions(self, image: np.ndarray) -> List:
        """Function to get the predictions from existing trackers.

        Args:
            image (np.ndarray): Original image in [H, W, 3] shape.

        Returns:
            List: List with each item having one tracking id and tracked bbox.
                Tracked bbox in [x1, y1, w, h] format.
        """
        result = []
        for id, tracker in self.all_trackers:
            status, bbox = tracker.update(image)
            result.append([id, bbox])

        return result
