##
# @author Meet Patel <patelmeet2012@gmail.com>
# @config.py File to store various configuration parameters.
# @desc Created on 2022-11-15 6:29:26 pm
# @copyright APPI SASU
##

# Detection
detection_iou_threshold = 0.65
detection_score_threshold = 0.5
class_mapping_file = "class_mapping.txt"
model_path = "../training/exported_models/yolov5m.onnx"
input_name = "images"
output_name = "output"


# Tracking
tracker_iou_threshold = 0.5
frame_interval = 10

# Det
