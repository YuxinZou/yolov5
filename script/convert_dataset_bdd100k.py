import os
import cv2
import json
import numpy as np
from tqdm import tqdm


def draw_label_type(draw_img, bbox, label, label_color):
    label = str(label)
    bbox = list(map(int, bbox))
    box_color = (255, 0, 255)
    cv2.rectangle(draw_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=box_color, thickness=2)
    labelSize = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
    if bbox[1] - labelSize[1] - 3 < 0:
        cv2.rectangle(draw_img,
                      (bbox[0], bbox[1] + 2),
                      (bbox[0] + labelSize[0],
                       bbox[1] + labelSize[1] + 3),
                      color=label_color,
                      thickness=-1)
        cv2.putText(draw_img, label,
                    (bbox[0], bbox[1] + labelSize[1] + 3),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    thickness=1)
    else:
        cv2.rectangle(draw_img,
                      (bbox[0], bbox[1] - labelSize[1] - 3),
                      (bbox[0] + labelSize[0], bbox[1] - 3),
                      color=label_color,
                      thickness=-1)
        cv2.putText(draw_img, label,
                    (bbox[0], bbox[1] - 3),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    thickness=1)
    return draw_img


def check_bdd(save_small=False):
    CLASS_NAMES = ["bus", ]  # "truck", 'rider', "person", "car", ]
    image_path = "../dataset/peoplecar/images/train/bdd10k/train"
    label_path = "../dataset/peoplecar/images/train/bdd10k/labels/train"
    for i in tqdm(os.listdir(image_path)):
        fname = os.path.join(image_path, i)
        label_json = os.path.join(os.path.join(label_path, i.replace('jpg', 'json')))
        if not os.path.exists(label_json):
            objects = []
        else:
            with open(label_json, 'r') as load_f:
                info = json.load(load_f)
                objects = info['frames'][0]['objects']
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_labels_ignore = []
        for obj in objects:
            if obj['category'] in CLASS_NAMES:
                class_index = CLASS_NAMES.index(obj['category'])
            else:
                class_index = -1

            if obj['category'] not in CLASS_NAMES:
                continue

            x1 = int(obj['box2d']['x1'])
            y1 = int(obj['box2d']['y1'])
            x2 = int(obj['box2d']['x2'])
            y2 = int(obj['box2d']['y2'])

            x_min = np.min((x1, x2))
            x_max = np.max((x1, x2))
            y_min = np.min((y1, y2))
            y_max = np.max((y1, y2))

            if class_index != -1:
                gt_bboxes.append([x_min, y_min, x_max, y_max])
                gt_labels.append(CLASS_NAMES[class_index])
            else:
                gt_bboxes_ignore.append([x_min, y_min, x_max, y_max])
                gt_labels_ignore.append('-1')

        img = cv2.imread(fname)
        if len(gt_bboxes):
            for gt_bbox, gt_label in zip(gt_bboxes, gt_labels):
                draw_label_type(img, gt_bbox, gt_label, label_color=(0, 0, 255))
            cv2.imwrite(os.path.join(f'../gt_plot/bdd/truck/{i}'), img)


def check_7class_car_people():
    CLASS_NAMES = ['3', ]  # '0', '1',
    path = "../dataset/peoplecar/images/train/7class_car_people"
    for root, dirs, files in os.walk(path):
        for name in tqdm(files):
            if name.endswith('jpg'):
                fname = os.path.join(root, name)
                json_file = fname.replace('jpg', 'json')
            else:
                continue

            if not os.path.exists(json_file):
                objects = []
            else:
                with open(json_file, 'r') as load_f:
                    info = json.load(load_f)
                    objects = info['shapes']
            gt_bboxes = []
            gt_labels = []
            gt_bboxes_ignore = []
            gt_labels_ignore = []

            for obj in objects:
                if obj['label'] in CLASS_NAMES:
                    class_index = CLASS_NAMES.index(obj['label'])
                else:
                    class_index = -1

                if obj['label'] not in CLASS_NAMES:
                    continue

                if obj['shape_type'] == "rectangle":
                    x1 = obj['points'][0][0]
                    y1 = obj['points'][0][1]
                    x2 = obj['points'][1][0]
                    y2 = obj['points'][1][1]
                    x_min = np.min((x1, x2))
                    x_max = np.max((x1, x2))
                    y_min = np.min((y1, y2))
                    y_max = np.max((y1, y2))

                    if class_index != -1:
                        gt_bboxes.append([x_min, y_min, x_max, y_max])
                        gt_labels.append(CLASS_NAMES[class_index])
                    else:
                        gt_bboxes_ignore.append([x_min, y_min, x_max, y_max])
                        gt_labels_ignore.append('-1')

            img = cv2.imread(fname)
            if len(gt_bboxes):
                for gt_bbox, gt_label in zip(gt_bboxes, gt_labels):
                    img = draw_label_type(img, gt_bbox, gt_label, label_color=(0, 0, 255))
                cv2.imwrite(os.path.join(f'../gt_plot/real/truck/{name}'), img)


if __name__ == '__main__':
    check_bdd()
