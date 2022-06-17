import glob
import json
import os

import cv2
import numpy as np
from tqdm import tqdm
import shutil


def convert(mode='val'):
    count = 0
    imagefolder = os.path.join('images', mode)
    labelfolder = imagefolder.replace('images', 'labels')
    os.makedirs(labelfolder, exist_ok=True)
    fi = open(f'{mode}.txt', 'w')
    for root, dirnames, filenames in os.walk(imagefolder):
        for filename in tqdm(filenames):
            if filename.endswith('jpg'):
                fname = os.path.join(root, filename)
                name = '/'.join(fname.split('\\'))
                fi.write(f'datasets/peoplecar/{name}' + '\n')
                # label_subfolder = root.replace('images', 'labels')
                # os.makedirs(label_subfolder, exist_ok=True)
                #
                # label_json = fname.replace('jpg', 'json')
                # if os.path.isfile(label_json):
                #     img = cv2.imread(fname)
                #     h, w, _ = img.shape
                #     label_txt = os.path.join(label_subfolder,
                #                              filename.replace('jpg', 'txt'))
                #     json2txt(label_json, label_txt, h, w)
                #     count += 1

    fi.close()
    print(f'txt total: {count}')


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
    img = 'images/train/7class_car_people/1025add_fix/D-07160079-XW/image_00001.jpg'
    txt = 'labels/train/7class_car_people/1025add_fix/D-07160079-XW/image_00001.txt'
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


def extract_bdd10k(mode='train'):
    root = '/mnt/yfs/sharedir/industrial/PUBLIC/detection/DataSets/BDD100K/bdd100k/'
    image_folder = os.path.join(root, f'images/10k/{mode}')
    for i in os.listdir(image_folder):
        fname = os.path.join(image_folder, i)
        label_file = os.path.join(root, f'labels/100k/{mode}',
                                  i.replace('jpg', 'json'))
        # assert os.path.isfile(label_file), f'not file {label_file}'
        # shutil.copy(label_file, f'./bdd10/labels/train')


def check_bdd10k(mode='train'):
    CLASS_NAMES = ["person", "car", "bus", "truck", 'rider', ]
    image_folder = f'../dataset/peoplecar/images/train/bdd10k/{mode}'
    label_folder = f'../dataset/peoplecar/images/train/bdd10k/labels/{mode}'
    cls_set = set()
    for i in tqdm(os.listdir(image_folder)):
        fname = os.path.join(image_folder, i)
        label_file = os.path.join(label_folder, i.replace('jpg', 'json'))
        if not os.path.exists(label_file):
            objects = []
        else:
            with open(label_file, 'r') as load_f:
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

            # if obj['attributes']['occluded'] or obj['attributes']['truncated']:
            #     continue
            print(obj)
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
                gt_bboxes_ignore.append(
                    [x_min, y_min, x_max, y_max])
                gt_labels_ignore.append('-1')

        img = cv2.imread(fname)
        if len(gt_bboxes):
            for gt_bbox, gt_label in zip(gt_bboxes, gt_labels):
                draw_label_type(img, gt_bbox, gt_label, label_color=(0, 0, 255))
            cv2.imshow('f', img)
            cv2.waitKey()


def json2txt(json_file, txt_file, h, w):
    CLASS_NAMES = ["person", "car", "bus", "truck", 'rider', ]
    with open(json_file, 'r') as load_f:
        info = json.load(load_f)
        objects = info['frames'][0]['objects']
    gt_bboxes = []
    gt_labels = []
    gt_bboxes_ignore = []
    gt_labels_ignore = []

    for obj in objects:
        if obj['category'] == 'rider':
            obj['category'] = 'person'

        if obj['category'] in CLASS_NAMES:
            class_index = CLASS_NAMES.index(obj['category'])
        else:
            class_index = -1

        if obj['category'] not in CLASS_NAMES:
            continue

        # if obj['attributes']['occluded'] or obj['attributes']['truncated']:
        #     continue
        x1 = int(obj['box2d']['x1'])
        y1 = int(obj['box2d']['y1'])
        x2 = int(obj['box2d']['x2'])
        y2 = int(obj['box2d']['y2'])

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


def generate_bdd10k_yolo():
    root = 'images/train/bdd10k'

    os.makedirs('labels/train/bdd10k/train', exist_ok=True)
    os.makedirs('labels/train/bdd10k/val', exist_ok=True)
    image_folder = root
    label_folder = os.path.join(root, 'labels')
    images = glob.glob(f"{image_folder}/*/*.jpg")
    print(len(images))
    fi = open(f'bdd10k.txt', 'w')
    for i in tqdm(images):
        print(i)
        fi.write(f'datasets/peoplecar/{i}' + '\n')
        json_file = os.path.join(label_folder,
                                 '/'.join(i.split('/')[-2:]).replace('jpg',
                                                                     'json'))
        assert os.path.isfile(json_file), f'json_file'
        # img = cv2.imread(i)
        # h, w, _ = img.shape
        # txt_file = os.path.join('labels/train/bdd10k/images',
        #                         '/'.join(i.split('/')[-2:]).replace('jpg',
        #                                                             'txt'))
        # json2txt(json_file, txt_file, h, w)
    fi.close()


def generate_bdd100k_yolo(mode='train'):
    root = f'images/{mode}/bdd100k/{mode}'

    os.makedirs(f'labels/{mode}/bdd100k/{mode}', exist_ok=True)
    image_folder = root
    label_folder = f'/mnt/yfs/sharedir/industrial/PUBLIC/detection/DataSets/BDD100K/bdd100k/labels/100k/{mode}'
    images = glob.glob(f"{image_folder}/*.jpg")
    print(len(images))
    fi = open(f'train_bdd100k.txt', 'w')
    for i in tqdm(images):
        print(i)
        fi.write(f'datasets/peoplecar/{i}' + '\n')
        i = i.split('/')[-1]
        json_file = os.path.join(label_folder, i.replace('jpg', 'json'))
        print(json_file)
        assert os.path.isfile(json_file), f'json_file: {json_file}'
        img = cv2.imread(os.path.join(image_folder, i))
        h, w, _ = img.shape
        txt_file = os.path.join(f'labels/{mode}/bdd100k/{mode}', i.replace('jpg', 'txt'))
        print(txt_file)
        # json2txt(json_file, txt_file, h, w)
    fi.close()


if __name__ == '__main__':
    check_bdd10k()
