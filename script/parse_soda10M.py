import json
import cv2
from abc import abstractmethod, ABCMeta
import numpy as np
from collections import defaultdict
import os
from tqdm import tqdm


class BaseParser(metaclass=ABCMeta):
    """The base class of parser, a help for data parser.
    All subclasses should implement the following APIs:v
    - ``__call__()``
    Args:
        imgs_folder (str, optional): Path of folder for Images. Default: ''.
        txt_file (str, optional): Required image paths. Default: None.
            Examples:
                xxx.jpg
                xxx.jpg
                xxxx.jpg
        extensions (str): Image extension. Default: 'jpg'.
    """

    def __init__(self,
                 imgs_folder='',
                 txt_file=None,
                 extension='jpg',
                 ):
        self.imgs_folder = imgs_folder
        self.txt_file = txt_file
        self.extension = extension

        self.imgs_list = None

        self._result = dict(img_names=None,
                            categories=None,
                            shapes=None,
                            bboxes=None,
                            labels=None,
                            segs=None,
                            scores=None,
                            bboxes_ignore=None,
                            labels_ignore=None, )

    @property
    def result(self):
        return self._result

    @result.setter
    def result(self, value):
        self._result.update(value)

    @staticmethod
    def _get_shape(fname):
        """Get image size.
        Args:
            fname (str): Absolute path of image.
        Returns:
            tuple: Image size.
        """

        img = cv2.imread(fname)
        return img.shape[0:2]

    @abstractmethod
    def __call__(self, need_shape):
        """Parse dataset.
        Args:
            need_shape (bool): Whether need shape attribute.
        Returns:
            dict: Annotations.
        """

        return self.result


class COCOParser(BaseParser):
    """Class of parser for COCO data format.
    Args:
        anno_path (str): Path of annotation file.
        ignore (bool): If set True, some qualified annotations will be ignored.
        min_size (int or float, optional): The minimum size of bounding
            boxes in the images. If the size of a bounding box is less than
            ``min_size``, it would be add to ignored field.
    """

    def __init__(self,
                 anno_path,
                 ignore=True,
                 min_size=None,
                 **kwargs):
        super(COCOParser, self).__init__(**kwargs)

        self.ignore = ignore
        self.min_size = min_size

        self.data = json.load(open(anno_path, 'r'))
        self.categories = [cat['name'] for cat in self.data['categories']]
        self.cat_ids = [cat['id'] for cat in self.data['categories']]
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}

        self.img2anns = defaultdict(list)
        for ann in self.data['annotations']:
            self.img2anns[ann['image_id']].append(ann)

        self.imgid2filename = dict()
        for img in self.data['images']:
            self.imgid2filename[img['id']] = img['file_name']

    def _check_ignore(self, ann):
        """Check whether the box needs to be ignored or not.
        Args:
            ann: Annotation of a box.
        """

        return ann.get('ignore', False) or \
               ann.get('iscrowd', False)

    def __call__(self, need_shape=True):
        fname_list, shapes_list, bboxes_list, labels_list, segs_list, \
        bboxes_ignore_list, labels_ignore_list = [], [], [], [], [], [], []
        for img in self.data['images']:
            if self.imgs_list is not None and \
                    img['file_name'] not in self.imgs_list:
                continue

            img_id = img['id']
            fname = os.path.join(self.imgs_folder, img['file_name'])
            if img.get('width') and img.get('height'):
                height, width = img['height'], img['width']
            else:
                height, width = self._get_shape(fname) if need_shape else (0, 0)

            ann_info = [ann for ann in self.img2anns[img_id]]

            bboxes, labels, segs, bboxes_ignore, labels_ignore = [], [], [], [], []
            for i, ann in enumerate(ann_info):
                ignore = self.ignore and self._check_ignore(ann)
                x1, y1, w, h = ann['bbox']
                inter_w = max(0, min(x1 + w, width) - max(x1, 0))
                inter_h = max(0, min(y1 + h, height) - max(y1, 0))
                if inter_w * inter_h == 0:
                    continue
                if ann['area'] <= 0 or w < 1 or h < 1:
                    continue
                if ann['category_id'] not in self.cat_ids:
                    continue
                bbox = list(map(float, [x1, y1, x1 + w, y1 + h]))
                if ignore:
                    bboxes_ignore.append(bbox)
                    labels_ignore.append(self.cat2label[ann['category_id']])
                else:
                    bboxes.append(bbox)
                    labels.append(self.cat2label[ann['category_id']])
                    segs.append(ann.get('segmentation', []))  # TODO: Default value of segmentation.

            if not bboxes:
                bboxes = np.zeros((0, 4))
                labels = np.zeros((0,))
            else:
                bboxes = np.array(bboxes)
                labels = np.array(labels)

            if not bboxes_ignore:
                bboxes_ignore = np.zeros((0, 4))
                labels_ignore = np.zeros((0,))
            else:
                bboxes_ignore = np.array(bboxes_ignore)
                labels_ignore = np.array(labels_ignore)

            fname_list.append(fname)
            shapes_list.append([width, height])
            bboxes_list.append(bboxes)
            labels_list.append(labels)
            bboxes_ignore_list.append(bboxes_ignore)
            labels_ignore_list.append(labels_ignore)
            segs_list.append(np.array(segs))

        self.result = dict(
            img_names=fname_list,
            categories=self.categories,
            shapes=shapes_list,
            bboxes=bboxes_list,
            labels=labels_list,
            segs=segs_list,
            bboxes_ignore=bboxes_ignore_list,
            labels_ignore=labels_ignore_list,
            imgid2filename=self.imgid2filename,
        )

        return self.result


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
    label_set = set()
    anno_path = 'D:/data/自动驾驶公开数据集/SODA10M/SSLAD-2D/labeled/annotations/instance_val.json'
    image_folder = 'D:/data/自动驾驶公开数据集/SODA10M/SSLAD-2D/labeled/val'
    cates = ['Pedestrian', 'Cyclist', 'Car', 'Truck', 'Tram', 'Tricycle']
    parser = COCOParser(anno_path)
    res = parser()
    print(res['categories'])
    for fname, bbox, label in zip(res['img_names'], res['bboxes'], res['labels']):
        print(fname)
        img = read_ch_img(os.path.join(image_folder, fname))
        if len(bbox):
            for gt_bbox, gt_label in zip(bbox, label):
                label_set.add(gt_label)
                print(gt_label)
                if gt_label in [0, 1, 2, 3, 4, 5]:
                    draw_label_type(img, gt_bbox, gt_label, label_color=(0, 0, 255))
            # cv2.imwrite(os.path.join(f'../gt_plot/bdd/truck/{i}'), img)
        cv2.imshow('f', img)
        cv2.waitKey()


def convert():
    fi = open(f'soda10M.txt', 'w')
    for sub in ['train', 'val']:
        anno_path = f'D:/data/自动驾驶公开数据集/SODA10M/SSLAD-2D/labeled/annotations/instance_{sub}.json'
        image_folder = f'D:/data/自动驾驶公开数据集/SODA10M/SSLAD-2D/labeled/{sub}'

        dst = 'D:/code/github/yolov5/dataset/qianhai_clean/images/train/soda10M'
        dst_label = dst.replace('images', 'labels')
        os.makedirs(dst, exist_ok=True)
        os.makedirs(dst_label, exist_ok=True)
        parser = COCOParser(anno_path)
        res = parser()
        print(res['categories'])

        for fname, bbox, label in tqdm(zip(res['img_names'], res['bboxes'], res['labels'])):
            fi.write(f'datasets/qianhai_clean/images/train/soda10M/{fname}' + '\n')
            img = read_ch_img(os.path.join(image_folder, fname))

            label_txt = fname.replace('jpg', 'txt')
            label_txt = os.path.join(dst_label, label_txt)

            img = read_ch_img(os.path.join(image_folder, fname))
            h, w, _ = img.shape

            gt_bboxes = []
            gt_labels = []
            # ['Pedestrian', 'Cyclist', 'Car', 'Truck', 'Tram', 'Tricycle']
            for box, l in zip(bbox, label):
                if l in [0, 1, 5]:
                    l = 0
                elif l == 2:
                    l = 1
                elif l == 3:
                    l = 3
                elif l == 4:
                    l = 2
                else:
                    continue
                x1 = box[0]
                y1 = box[1]
                x2 = box[2]
                y2 = box[3]
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

            if len(gt_bboxes):
                f = open(label_txt, 'w')
                for gt_bbox, gt_label in zip(gt_bboxes, gt_labels):
                    f.write(str(gt_label) + " " + " ".join(
                        [str(a) for a in gt_bbox]))
                    f.write('\n')
                f.close()

    fi.close()

def check_yolo_format():
    folder = 'D:/code/github/yolov5/dataset/qianhai_clean/labels/train/soda10M'
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
