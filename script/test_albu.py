import albumentations as A
import cv2

from einops import rearrange, reduce, repeat


def test_albu():
    p = 0.5
    transform = A.Compose([
        A.Blur(p=p),
        A.GaussNoise(p=p),
        A.ToGray(p=0.01),
        A.CLAHE(p=0.01),
    ])

    # Read an image with OpenCV and convert it to the RGB colorspace
    image = cv2.imread("../dataset/peoplecar/images/train/7class_car_people/1025add_fix/D-07011221-XW/image_00001.jpg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Augment an image
    for i in range(100):
        transformed = transform(image=image)
        transformed_image = transformed["image"]
        transformed_image = cv2.cvtColor(transformed_image, cv2.COLOR_RGB2BGR)
        cv2.imshow('f', transformed_image)
        cv2.waitKey()


import numpy as np


def read_ch_img(fname):
    return cv2.imdecode(np.fromfile(fname, dtype=np.uint8), cv2.IMREAD_COLOR)


def test_roi():
    from shapely.geometry import Point, Polygon
    fname = 'D:/data/前海/测试集图片/qianhai_1/image_00000004.jpg'
    img = read_ch_img(fname)
    print(img.shape)
    roi = {
        # 0: [(500, 200), (800, 1080), (200, 1080)],
        # 1: [(1120, 100), (1420, 300), (1720, 200), (1720, 1080), (900, 900)]
        2: [(0, 0), (200, 0), (1920, 880), (1920, 1080), (1720, 1080), (0, 200)],
    }

    print(len(roi))
    roi_poly = dict()
    for k, v in roi.items():
        roi_poly.update({k: Polygon(v)})

    p1 = Point(500, 600)
    p2 = Point(1500, 700)

    for roi_id, poly in roi_poly.items():
        print([np.array(roi[roi_id], dtype=np.int32)])
        cv2.polylines(img, [np.array(roi[roi_id], dtype=np.int32)], 1, (255, 0, 0), thickness=2)
        print(poly.contains(p1))
        print(poly.contains(p2))
    cv2.imshow('f', img)
    cv2.waitKey()


def random_color():
    _COLOR_FP = (0, 0, 1.0)
    _COLOR_FN = (1.0, 0, 0)
    import random
    color_map = dict()
    color_map['fp'] = _COLOR_FP
    color_map['fn'] = _COLOR_FN
    random.seed(0)
    for i in range(0, 80):
        color_map.update({i: (random.randint(0, 200) / 255.0,
                              random.randint(0, 200) / 255.0,
                              random.randint(0, 200) / 255.0)})
    return color_map


if __name__ == "__main__":
    test_roi()
