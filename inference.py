from utils.augmentations import letterbox
from utils.torch_utils import select_device
from utils.plots import Annotator, colors
from utils.general import (
    check_img_size, increment_path, non_max_suppression, scale_coords, resizeImage, overlay, draw_results)
from utils.datasets import LoadImages, LoadStreams
from models.common import DetectMultiBackend
from pathlib import Path

import cv2
import torch

import os
import sys
import CONFIG as cfg
import numpy as np
import time
from tensorflow.keras.models import load_model
import tensorflow as tf
import pandas as pd
from datetime import datetime

# import deepsort
from deep_sort.tracker import Tracker
from deep_sort.detection import Detection
from deep_sort import nn_matching, generate_detections as gdet
from deep_sort.utils import format_boxes


def detectLogo(logo, imgVeh, device, model_logo, imgsz_logo):
    im0 = imgVeh.copy()
    img = letterbox(imgVeh, imgsz_logo, stride=int(
        model_logo.stride), auto=True)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    img = img.float()  # uint8 to fp16/32
    img = img / 255.0  # 0 - 255 to 0.0 - 1.0
    if len(img.shape) == 3:
        img = img[None]  # expand for batch dim

    # Inference
    pred = model_logo(img, augment=logo.augment, visualize=logo.visualize)
    # NMS: select the first detection as there can only be one License Plate per vehicle
    det = non_max_suppression(pred, logo.conf_thres, logo.iou_thres,
                              logo.classes, logo.agnostic_nms, max_det=logo.max_det)[0]
    brands_predicted = []
    confidences = []
    # for i, det in enumerate(pred):
    if det is not None and len(det):
        # Rescale boxes from img_size to im0 size
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
        for *box, conf, cls in reversed(det):
            # annotate vehicle labels
            brands_predicted.append(model_logo.names[int(cls)])
            confidences.append(float(conf))
    b = brands_predicted[np.argmax(confidences)] if len(
        confidences) > 0 else ''
    return b


@torch.no_grad()
def run(vhl, logo):

    # press space to pause/play the video while inference
    stop = False

    # Directories
    save_dir = increment_path(os.path.join(cfg.OUTPUT_PATH, cfg.EXP_NAME))  # increment run
    (save_dir / 'crops' if cfg.SAVE_CROPS else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # initialize deep sort
    max_cosine_distance = 0.4
    nn_budget = None
    encoder = gdet.create_box_encoder(cfg.DEEPSORT_WEIGHTS_PATH, batch_size=1)
    # calculate cosine distance metric
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    # initialize tracker
    tracker = Tracker(metric)

    # Initialize and Load car detection model
    device = select_device(cfg.DEVICE)
    model = DetectMultiBackend(vhl.weights, device=device, dnn=False)
    stride, names, pt, jit, onnx = model.stride, model.names, model.pt, model.jit, model.onnx
    imgsz = check_img_size(vhl.imgsz, s=stride)  # check image size
    model.model.float()
    model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.model.parameters())))  # warmup

    dataset = LoadImages(cfg.SOURCE, img_size=imgsz, stride=stride, auto=pt and not jit)
    bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Initialize and Load license plate detection model
    model_logo = DetectMultiBackend(logo.weights, device=device, dnn=False)
    imgsz_logo = check_img_size(logo.imgsz, s=model_logo.stride)  # check image size
    model_logo.model.float()
    model_logo(torch.zeros(1, 3, *imgsz_logo).to(device).type_as(next(model_logo.model.parameters())))  # warmup

    color_clf = load_model(cfg.COLOR_CLF)
    color_clf = tf.keras.Sequential([color_clf, tf.keras.layers.Softmax()])

    t0 = time.time()
    vehicle_counter = 0
    color_to_show = 'n/a'
    brand_to_show = 'n/a'
    type_to_show = 'n/a'
    tracked_ids = dict()
    processed_ids = []

    df = pd.DataFrame(columns=['time instance', 'id', 'type', 'color', 'brand'])
    for path, im, im0s, vid_cap, s in dataset:
        im = torch.from_numpy(im).to(device)
        im = im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        # draw area rectangle mask
        im0 = overlay(cfg, im0s.copy(), alpha=0.2, color=(0, 255, 0))

        # Inference
        pred = model(im, augment=vhl.augment, visualize=False)
        # NMS
        pred = non_max_suppression(pred, vhl.conf_thres, vhl.iou_thres, vhl.classes,
                                   vhl.agnostic_nms, max_det=vhl.max_det)
        # Process predictions
        bboxes, scores, labels = [], [], []
        for i, det in enumerate(pred):  # detections per image
            imc = im0s.copy()

            p = Path(path)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            annotator = Annotator(im0, line_width=vhl.line_thickness, example=str(names))

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class

                 # Write results
                for *xyxy, conf, cls in reversed(det):
                    x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                    # counting vehicles
                    center = (int((x1 + x2) // 2), int((y1 + y2) // 2))

                    # if a vehicle is within the area of interest, then condider the box for tracking
                    if im0.shape[0] * cfg.AREA_THRESHOLD[0] < center[1] < im0.shape[0] * cfg.AREA_THRESHOLD[1]:
                        bboxes.append(xyxy)
                        scores.append(conf)
                        labels.append(names[int(cls)])

                # format bounding boxes from normalized xmin, ymin, xmax, ymax ---> xmin, ymin, width, height
                bboxes = format_boxes(bboxes)
                # encode yolo detections and feed to tracker
                features = encoder(im0, bboxes)
                detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, labels, features)]
                # Call the tracker
                tracker.predict()
                tracker.update(detections)

                # update tracks
                for track in tracker.tracks:
                    if not track.is_confirmed() or track.time_since_update > 1:
                        continue

                    # get tracked person ids and boxes
                    bbox = track.to_tlbr()
                    label = track.get_class()
                    track_id = track.get_id()

                    # continue if a vehicle value  has been entered in record already
                    if track_id in processed_ids:
                        continue

                    # initialize tracking vehicle if it's the first time 
                    if track_id not in tracked_ids:
                        tracked_ids[track_id] = {
                            "colors": [],
                            "brands": [],
                            "types": [],
                            "appearance": 0
                        }

                    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                    # counting vehicles
                    center = (int((x1 + x2) // 2), int((y1 + y2) // 2))
                    cv2.circle(im0, center, 4, (0, 0, 255), -1)

                    brand, color = '', ''
                    imgVeh = imc[y1:y2, x1:x2]

                    if cfg.DETECT_BRAND:
                        try:
                            brand = detectLogo(logo, imgVeh.copy(), device, model_logo, imgsz_logo)
                        except:
                            ...

                    # predicting the color of the vehicle
                    if cfg.DETECT_COLOR:
                        try:
                            color_img = imgVeh.copy()
                            color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB) # convert to RGB
                            color_img = cv2.resize(color_img, cfg.COLOR_CLF_SIZE)
                            color_pred = color_clf.predict(np.expand_dims(color_img, axis=0))[0]
                            color = cfg.COLOR_CLASSES[
                                np.argmax(color_pred)
                            ]
                        except:
                            ...

                    # save vehicle information against specific tracking id
                    tracked_ids[track_id]["colors"].append(color)
                    tracked_ids[track_id]["brands"].append(brand)
                    tracked_ids[track_id]["types"].append(label)
                    tracked_ids[track_id]["appearance"] += 1

                    # annotate vehicle labels within area
                    c = names.index(label)
                    label_ = f'{track_id}: {label} {brand} {color}'
                    annotator.box_label([x1, y1, x2, y2], label_, color=colors(c, True))

                    # if vehicle is about to leave the area of intereset, then process it
                    if y2 >= im0.shape[0] * cfg.AREA_THRESHOLD[1]:
                        # check if a certain vehicle has appeared more then 5 frames
                        if tracked_ids[track_id]["appearance"] > 5:
                            vehicle_counter += 1    # increase the vehicle count
                            # select the most frequent detection for color, type and brand
                            colors_ = [c for c in tracked_ids[track_id]["colors"] if c != '']
                            color_to_show = max(set(colors_), key = colors_.count) if len(colors_) > 0 else 'n/a'

                            brands = [t for t in tracked_ids[track_id]["brands"] if t != '']
                            brand_to_show = max(set(brands), key = brands.count) if len(brands) > 0 else 'n/a'

                            types = [t for t in tracked_ids[track_id]["types"] if t != '']
                            type_to_show = max(set(types), key = types.count) if len(types) > 0 else 'n/a'

                            # append the vehicle record to csv
                            df = df.append({
                                'time instance': datetime.now(), 
                                'id': track_id,
                                'type': type_to_show,
                                'color': color_to_show,
                                'brand': brand_to_show
                                }, ignore_index=True)

                        del tracked_ids[track_id]
                        processed_ids.append(track_id)

                    # saving crops
                    if cfg.SAVE_CROPS:
                        crops_path = save_dir / 'crops'
                        if brand != '' and imgVeh.shape[0] > 40 and imgVeh.shape[1] > 40:
                            crop_path = f"{crops_path}/{p.name[:-4]}_{datetime.now().strftime('%y%m%d_%H%M%S')}.jpg"
                            cv2.imwrite(crop_path, imgVeh)

        # draw results on frame
        im0 = draw_results(im0, vehicle_counter, type_to_show, color_to_show, brand_to_show)
        im0 = annotator.result()

        # Save results (image with detections)
        if cfg.SAVE_IMG:
            if dataset.mode == 'image':
                cv2.imwrite(save_path, im0)
            else:  # 'video' or 'stream'
                if vid_path[i] != save_path:  # new video
                    vid_path[i] = save_path
                    if isinstance(vid_writer[i], cv2.VideoWriter):
                        vid_writer[i].release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path += '.mp4'
                    vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer[i].write(im0)

        if cfg.VIS:
            # resize the image for better visualization
            im0 = resizeImage(im0, width=1440)
            cv2.imshow("detection", im0)
            if cv2.waitKey(0 if stop else 1) == 32:
                stop = not stop

        # export results to csv
        df.to_csv(os.path.join(save_dir, "result.csv"), index=False)

    print(f'Done...')
    cv2.destroyAllWindows()


if __name__ == "__main__":
    vhl = cfg.Vehicle()
    logo = cfg.Logo()
    run(vhl, logo)
