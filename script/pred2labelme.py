import os
import re
import json

import cv2
import numpy as np
from tqdm import tqdm
import shutil


def read_ch_img(fname):
    return cv2.imdecode(np.fromfile(fname, dtype=np.uint8), cv2.IMREAD_COLOR)


def find_unchinese(file):
    pattern = re.compile(r'[\u4e00-\u9fa5]')
    unchinese = re.sub(pattern, "", file)
    return unchinese


def clean(string):
    string = string.replace('（', '')
    string = string.replace('）', '')
    return string


def pred2labelme(img_folder, label_folder, json_folder):
    version = '4.5.10'
    flags = {}
    for i in tqdm(os.listdir(img_folder)):
        dic = {}
        dic['version'] = version
        dic['flags'] = flags
        dic['shapes'] = []
        img = read_ch_img(os.path.join(img_folder, i))
        imageHeight, imageWidth, _ = img.shape
        txt_path = os.path.join(label_folder, i.replace('jpg', 'txt'))
        if not os.path.isfile(txt_path):
            continue
        file_obj = open(txt_path, encoding="gbk")
        all_lines = file_obj.readlines()
        for line in all_lines:
            shape = {}
            data = line.strip().split(' ')
            shape['label'] = data[0]
            x = float(data[1]) * imageWidth
            y = float(data[2]) * imageHeight
            w = float(data[3]) * imageWidth
            h = float(data[4]) * imageHeight
            x1 = x - w / 2
            y1 = y - h / 2
            x2 = x1 + w
            y2 = y1 + h
            shape['points'] = [[x1, y1], [x2, y2]]
            shape['shape_type'] = 'rectangle'
            shape['flags'] = {}
            shape['group_id'] = None
            dic['shapes'].append(shape)
        dic['imagePath'] = find_unchinese(clean(i))
        dic['imageData'] = None
        dic['imageHeight'] = imageHeight
        dic['imageWidth'] = imageWidth
        fw = open(os.path.join(json_folder, find_unchinese(clean(i)).replace('jpg', 'json')), 'w')
        print(os.path.join(json_folder, find_unchinese(clean(i))))
        cv2.imwrite(os.path.join(json_folder, find_unchinese(clean(i))), img)
        json.dump(dic, fw, indent=2)
        file_obj.close()


def pred2labelmeall():
    root = 'D:/data/前海/new_16'
    for path, dir_list, file_list in os.walk(root):
        if len(dir_list) > 10:
            continue
        for dir_name in dir_list:
            img_folder = f'{path}/{dir_name}'
            json_folder = img_folder.replace('new_16', 'new_16_json')
            os.makedirs(json_folder, exist_ok=True)
            label_folder = os.path.join(subfolder, 'labels')





    # for i in tqdm(os.listdir(root)):
    #     subfolder = os.path.join(root, i)
    #     img_folder = os.path.join(subfolder, 'img_ori')
    #     label_folder = os.path.join(subfolder, 'labels')
    #     json_folder = os.path.join('../runs/detect/labelme', find_unchinese(clean(i)))
    #     os.makedirs(json_folder, exist_ok=True)
    #     print(json_folder)
    #     pred2labelme(img_folder, label_folder, json_folder)


def extract_partial_data():
    src = '../runs/detect/labelme'
    dst = '../runs/detect/labelme部分'

    for i in tqdm(os.listdir(src)):
        os.makedirs(os.path.join(dst, i), exist_ok=True)
        count = 0
        subfolder = os.path.join(src, i)
        imgs = [j for j in sorted(os.listdir(subfolder)) if j.endswith('jpg')]
        for img in imgs:
            count += 1
            if count == 5:
                count = 0
                shutil.copy(os.path.join(subfolder, img), os.path.join(dst, i))
                shutil.copy(os.path.join(subfolder, img.replace('jpg', 'json')), os.path.join(dst, i))


if __name__ == '__main__':
    pred2labelmeall()
