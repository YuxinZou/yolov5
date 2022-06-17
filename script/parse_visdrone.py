import cv2
import os
import numpy as np

image_folder = 'D:/data/自动驾驶公开数据集/VisDrone/VisDrone2019-DET-train/images'
json_folder = 'D:/data/自动驾驶公开数据集/VisDrone/VisDrone2019-DET-train/annotations'


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


def read_ch_img(fname):
    return cv2.imdecode(np.fromfile(fname, dtype=np.uint8), cv2.IMREAD_COLOR)

def show_data():

    class_name = ['ignored regions', 'pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle',
                  'awning-tricycle', 'bus', 'motor', 'others']
    need_cls = ['bicycle', 'van', 'truck', 'tricycle','awning-tricycle']  # 'pedestrian','people', 'car','van','truck', 'bus'
    for i in os.listdir(image_folder):
        fname = os.path.join(image_folder, i)
        img = read_ch_img(fname)
        txt_file = os.path.join(json_folder, i.replace('jpg', 'txt'))
        file_obj = open(txt_file, encoding="gbk")
        all_lines = file_obj.readlines()
        bboxes = []
        labels = []
        truncateds = []
        difficults = []
        for lines in all_lines:
            shape = {}
            line = lines.strip().split(',')
            print(line)
            label = class_name[int(line[5])]
            truncated = int(line[6])
            difficult = int(line[7])
            xyxy = [int(line[0]), int(line[1]), int(line[0]) + int(line[2]) - 1, int(line[1]) + int(line[3]) - 1]
            bboxes.append(xyxy)
            labels.append(label)
            if truncated > 0:
                truncateds.append(True)
            else:
                truncateds.append(False)
            if difficult > 0:
                difficults.append(True)
            else:
                difficults.append(False)

        if len(bboxes):
            for gt_bbox, gt_label, truncated, difficult in zip(bboxes, labels, truncateds, difficults):
                if gt_label in need_cls:
                    draw_label_type(img, gt_bbox, gt_label, label_color=(0, 0, 255))
            # cv2.imwrite(os.path.join(f'../gt_plot/bdd/truck/{i}'), img)
        cv2.imshow('f', img)
        cv2.waitKey()

def convert():
    class_name = ['ignored regions', 'pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle',
                  'awning-tricycle', 'bus', 'motor', 'others']
    fi = open(f'VisDrone.txt', 'w')
    for sub in ['train', 'val']:
        anno_path = f'D:/data/自动驾驶公开数据集/VisDrone/VisDrone2019-DET-{sub}/annotations'
        image_folder = f'D:/data/自动驾驶公开数据集/VisDrone/VisDrone2019-DET-{sub}/images'

        dst = 'D:/code/github/yolov5/dataset/qianhai_clean/images/train/VisDrone'
        dst_label = dst.replace('images', 'labels')
        os.makedirs(dst, exist_ok=True)
        os.makedirs(dst_label, exist_ok=True)

        for name in os.listdir(anno_path):
            label_file = os.path.join(anno_path, name)
            fname = os.path.join(image_folder, name.replace('txt', 'jpg'))
            fi.write(f"datasets/qianhai_clean/images/train/VisDrone/{name.replace('txt', 'jpg')}" + '\n')
            img = read_ch_img(fname)
            h, w, _ = img.shape

            file_obj = open(label_file, encoding="gbk")
            all_lines = file_obj.readlines()
            gt_bboxes = []
            gt_labels = []

            for lines in all_lines:
                line = lines.strip().split(',')
                print(line)
                l = int(line[5])
                if l in [1, 2]:
                    l = 0
                elif l in [4, 5]:
                    l = 1
                elif l == 6:
                    l = 3
                elif l == 9:
                    l = 2
                else:
                    continue

                x1 = int(line[0])
                y1 = int(line[1])
                x2 = int(line[0]) + int(line[2]) - 1
                y2 = int(line[1]) + int(line[3]) - 1
                x_min = np.max((np.min((x1, x2)), 0))
                x_max = np.min((np.max((x1, x2)), w))
                y_min = np.max((np.min((y1, y2)), 0))
                y_max = np.min((np.max((y1, y2)), h))
                x_center = (x_min + x_max) / 2 / w
                y_center = (y_min + y_max) / 2 / h
                width = (x_max - x_min) / w
                height = (y_max - y_min) / h
                gt_bboxes.append([x_center, y_center, width, height])
                gt_labels.append(l)

            label_txt = os.path.join(dst_label, name)
            if len(gt_bboxes):
                f = open(label_txt, 'w')
                for gt_bbox, gt_label in zip(gt_bboxes, gt_labels):
                    f.write(str(gt_label) + " " + " ".join(
                        [str(a) for a in gt_bbox]))
                    f.write('\n')
                f.close()

    fi.close()


def check_yolo_format():
    folder = 'D:/code/github/yolov5/dataset/qianhai_clean/labels/train/VisDrone'
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
    check_yolo_format()
