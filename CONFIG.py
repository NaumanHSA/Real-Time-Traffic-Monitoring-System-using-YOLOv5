import os

SOURCE = "test_videos/test_video.mp4"
# SOURCE = "test_image"

# we can stop/start brand and color detection
DETECT_BRAND = True
DETECT_COLOR = True

VIS = True
SAVE_CROPS = False   # save cropped prediction boxes
SAVE_IMG = True      # save images/video 

DEVICE = ''  # cuda device, i.e. 0 or 0,1,2,3 or cpu

OUTPUT_PATH = 'runs/detect'  # save results to project/name
EXP_NAME = 'exp'  # save results to project/name

# PATHS
# COLOR_CLF = "weights/gbc_color.pkl"
COLOR_CLF = "weights/color.h5"
COLOR_CLF_SIZE = (32, 32)

# COLOR_CLASSES = "data/colors.txt"
COLOR_CLASSES = ['black', 'blue', 'green', 'orange', 'red', 'white', 'yellow']

""" These must be adjusted according to video """

# the percentage of height of frame within which the models will make predictions
# AREA_THRESHOLD is defined so the models do not process the entire frame, but instead, a part of frame (ROI)
# this brings significant improvements in speed
AREA_THRESHOLD = [0.4, 1.0] # [start_y, end_y]
assert AREA_THRESHOLD[0] < AREA_THRESHOLD[1], "AREA_THRESHOLD: start should be less then the end"

# Path to pre-trained DeepSort weights
DEEPSORT_WEIGHTS_PATH = 'weights/mars-small128.pb'
assert os.path.exists(DEEPSORT_WEIGHTS_PATH), f"{DEEPSORT_WEIGHTS_PATH} doesn't exist"

class Vehicle:
    def __init__(self):
        # self.weights = "weights/car_detection_best.pt"
        self.weights = "weights/car_detection_best.pt"
        self.imgsz = [608]  # inference size (pixels)
        self.imgsz *= 2 if len(self.imgsz) == 1 else 1

        self.conf_thres = 0.7  # confidence threshold
        self.iou_thres = 0.4  # NMS IOU threshold
        self.max_det = 1000  # maximum detections per image

        self.classes = None  # filter by class: --class 0, or --class 0 2 3
        self.agnostic_nms = False  # class-agnostic NMS
        
        self.augment = False  # augmented inference
        self.visualize = False  # visualize features
        self.update = False  # update all models
        
        
        self.exist_ok = False  # existing project/name ok, do not increment
        self.line_thickness = 3  # bounding box thickness (pixels)
        self.hide_labels = False  # hide labels
        self.hide_conf = False  # hide confidences


class Logo:
    def __init__(self):
        self.weights = "weights/car_logo_best_13_classes.pt"
        self.imgsz = [416]  # inference size (pixels)
        self.imgsz *= 2 if len(self.imgsz) == 1 else 1

        self.conf_thres = 0.4  # confidence threshold
        self.iou_thres = 0.5  # NMS IOU threshold
        self.max_det = 1000  # maximum detections per image

        self.classes = None  # filter by class: --class 0, or --class 0 2 3
        self.agnostic_nms = False  # class-agnostic NMS
        
        self.augment = False  # augmented inference
        self.visualize = False  # visualize features
        self.update = False  # update all models
        
        
        self.exist_ok = False  # existing project/name ok, do not increment
        self.line_thickness = 2  # bounding box thickness (pixels)
        self.hide_labels = False  # hide labels
        self.hide_conf = False  # hide confidences
