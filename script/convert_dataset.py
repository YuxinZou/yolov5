import json
import os
import shutil

import cv2
import numpy as np
from tqdm import tqdm

CLASS_NAMES_1 = ['0', '1', '2', '3']


def label_txtjson2txt(json_file, txt_file, h, w):
    with open(json_file, 'r') as load_f:
        gt_json = json.load(load_f)
        objects = gt_json['shapes']
    gt_bboxes = []
    gt_labels = []
    gt_bboxes_ignore = []
    gt_labels_ignore = []
    for obj in objects:
        if not obj['shape_type'] == "rectangle":
            continue
        if obj['label'] == '6':
            obj['label'] = '0'
        if obj['label'] in CLASS_NAMES_1:
            class_index = CLASS_NAMES_1.index(obj['label'])
        else:
            class_index = -1

        if obj['shape_type'] == "rectangle":
            x1 = obj['points'][0][0]
            y1 = obj['points'][0][1]
            x2 = obj['points'][1][0]
            y2 = obj['points'][1][1]
            x_min = np.max((np.min((x1, x2)), 0))
            x_max = np.min((np.max((x1, x2)), w))
            y_min = np.max((np.min((y1, y2)), 0))
            y_max = np.min((np.max((y1, y2)), h))
            x_center = (x_min + x_max) / 2 / w
            y_center = (y_min + y_max) / 2 / h
            width = (x_max - x_min) / w
            height = (y_max - y_min) / h
            if class_index != -1:
                gt_bboxes.append([x_center, y_center, width, height])
                gt_labels.append(class_index)
            else:
                gt_bboxes_ignore.append([x_min, y_min, x_max, y_max])
                gt_labels_ignore.append(class_index)

            if len(gt_bboxes):
                f = open(txt_file, 'w')
                for gt_bbox, gt_label in zip(gt_bboxes, gt_labels):
                    f.write(str(gt_label) + " " + " ".join(
                        [str(a) for a in gt_bbox]))
                    f.write('\n')
                f.close()


def convert(mode='val'):
    count = 0
    path = f'../dataset/qianhai_clean/images/{mode}/add_20220616/'

    os.makedirs(f'../dataset/qianhai_clean/labels/{mode}/add_20220616/', exist_ok=True)
    fi = open(f'add_20220616_{mode}.txt', 'w')
    for root, dirs, files in os.walk(path):
        for filename in tqdm(files):
            if filename.endswith('jpg'):
                fname = os.path.join(root, filename)
                # print(fname)
                # print(root)
                json_file = fname.replace('jpg', 'json')
                name = '/'.join(fname.split('/')[2:])
                name = name.replace('\\', '/')
                name = f'datasets/{name}'
                print(name)
                fi.write(name + '\n')
                label_subfolder = root.replace('images', 'labels')
                os.makedirs(label_subfolder, exist_ok=True)

                label_json = fname.replace('jpg', 'json')
                if os.path.isfile(json_file):
                    label_txt = os.path.join(label_subfolder,
                                             filename.replace('jpg', 'txt'))
                    if os.path.isfile(label_txt):
                        pass
                    else:
                        img = cv2.imread(fname)
                        h, w, _ = img.shape
                        json2txt(label_json, label_txt, h, w)
                        count += 1

    fi.close()
    print(f'total txt: {count}')


def draw_label_type(draw_img, bbox, label, label_color):
    label = str(label)
    bbox = list(map(int, bbox))
    box_color = (255, 0, 255)
    cv2.rectangle(draw_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                  color=box_color,
                  thickness=2)
    labelSize = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[
        0]
    if bbox[1] - labelSize[1] - 3 < 0:
        cv2.rectangle(draw_img,
                      (bbox[0], bbox[1] + 2),
                      (bbox[0] + labelSize[0], bbox[1] + labelSize[1] + 3),
                      color=label_color,
                      thickness=-1
                      )
        cv2.putText(draw_img, label,
                    (bbox[0], bbox[1] + labelSize[1] + 3),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    thickness=1
                    )
    else:
        cv2.rectangle(draw_img,
                      (bbox[0], bbox[1] - labelSize[1] - 3),
                      (bbox[0] + labelSize[0], bbox[1] - 3),
                      color=label_color,
                      thickness=-1
                      )
        cv2.putText(draw_img, label,
                    (bbox[0], bbox[1] - 3),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    thickness=1
                    )

def check_txt():
    folder = '../../../data/qianhai_clean/labels/val/7class_car_people/add_1104/CZ-06-0039_001_2021-07-19-08-30-00_2021-07-19-09-00-02/'
    for txt in os.listdir(folder):
        txt = os.path.join(folder, txt)
        img = txt.replace('labels', 'images').replace('txt', 'jpg')
        gt_labels = []
        gt_bboxes = []
        img = cv2.imread(img)
        h, w, c = img.shape
        print(img.shape)
        f2 = open(txt, "r")
        lines = f2.readlines()
        # cls, x_c, y_c, w, h
        for line3 in lines:
            line3 = line3.strip().split(' ')
            print(line3)
            gt_labels.append(line3[0])
            x1 = int((float(line3[1]) - float(line3[3]) / 2) * w)
            x2 = int((float(line3[1]) + float(line3[3]) / 2) * w)
            y1 = int((float(line3[2]) - float(line3[4]) / 2) * h)
            y2 = int((float(line3[2]) + float(line3[4]) / 2) * h)
            gt_bboxes.append([x1, y1, x2, y2])
        print(gt_bboxes, gt_labels)
        for gt_bbox, gt_label in zip(gt_bboxes, gt_labels):
            draw_label_type(img, gt_bbox, gt_label, label_color=(0, 0, 255))
        cv2.imshow('f', img)
        cv2.waitKey()




if __name__ == '__main__':
    convert()
    # check_txt()
    # extract_bdd10k()
