##
# @author Meet Patel <patelmeet2012@gmail.com>
# @utils.py Miscellaneous utility functions file.
# @desc Created on 2022-11-15 8:02:01 pm
# @copyright APPI SASU
##

from typing import Dict, List, Tuple, Union

import cv2
import numpy as np


def resize_image(
    image: np.ndarray, dst_dims: Union[Tuple, List] = (640, 640)
) -> Tuple[np.ndarray, int, int, float]:
    """Function to resize the image by padding and scalling it to provided dims.

    Args:
        image (np.ndarray): RGB Image in format [H, W, 3]
        dst_dims (Union[Tuple, List], optional): Resize dimensions. Defaults to (640, 640).

    Returns:
        Tuple[np.ndarray, int, int, float]: Tuple of resized image, padding on
            top, padding on left, scalling value to scale image to resize dims.
    """
    image_h, image_w = image.shape[:2]
    img_ar = image_h / image_w

    dst_h, dst_w = dst_dims
    if img_ar < 1.0:
        new_w = dst_w
        new_h = int(new_w * img_ar)
        pad_h = (dst_h - new_h) // 2
        pad_w = 0
    else:
        new_h = dst_h
        new_w = int(new_h / img_ar)
        pad_w = (dst_w - new_w) // 2
        pad_h = 0

    scale = image_h / new_h

    resized_image = cv2.resize(image, (new_w, new_h))
    resized_image = cv2.copyMakeBorder(
        resized_image,
        pad_h,
        dst_h - new_h - pad_h,
        pad_w,
        dst_w - new_w - pad_w,
        cv2.BORDER_CONSTANT,
        128,
    )

    assert resized_image.shape == (640, 640, 3)
    return resized_image, pad_h, pad_w, scale


def plot_boxes(image: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Function to plot the bounding boxes on given image.

    Args:
        image (np.ndarray): Image in RGB format with [H, W, 3] dims.
        boxes (np.ndarray): Bounding boxes in [xc, yc, w, h, cls_id, cls_prob]
            format.

    Returns:
        np.ndarray: Image with boxes plotted on it.
    """
    for bb in boxes:
        xc, yc, w, h, cls_ids = [int(c) for c in bb[:5]]
        X1 = int(xc - w / 2)
        Y1 = int(yc - h / 2)
        X2 = X1 + w
        Y2 = Y1 + h
        score = bb[-1]
        print(
            f"X1={X1}, Y1={Y1}, X2={X2}, Y2={Y2}, Class-id={cls_ids}, " 
            f"Score={score*100:.3f}"
        )
        cv2.rectangle(image, (X1, Y1), (X2, Y2), (0, 255, 0), 2)
    return image


def display(image: np.ndarray) -> None:
    """Display image using cv2.

    Args:
        image (np.ndarray): Image with [H, W, 3] shape.
    """
    cv2.imshow("img", image)
    cv2.waitKey()
    cv2.destroyAllWindows()


def get_class_mapping_dict(mapping_file: str) -> Dict:
    """Function to parse the class mapping file and convert it into dict.

    Args:
        mapping_file (str): Class mapping file.

    Returns:
        Dict: Dict with mapping from class name to class id.
    """
    with open(mapping_file, "r") as file:
        lines = file.readlines()

    class_mapping = {}
    for i, line in enumerate(lines):
        line = line.replace("\n", "")
        class_mapping[line] = i

    return class_mapping
