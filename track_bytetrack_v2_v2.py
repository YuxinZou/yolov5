import argparse

import os

# limit the number of cpus used by high performance libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import numpy as np
from pathlib import Path
import torch
from scipy.interpolate import interp1d
from tqdm import tqdm
from typing import Tuple, List, Dict
from PIL import Image, ImageDraw, ImageFont

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 strongsort root directory
WEIGHTS = ROOT / 'weights'

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if str(ROOT / 'yolov5') not in sys.path:
    sys.path.append(str(ROOT / 'yolov5'))  # add yolov5 ROOT to PATH
if str(ROOT / 'strong_sort') not in sys.path:
    sys.path.append(str(ROOT / 'strong_sort'))  # add strong_sort ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import logging
from models.common import DetectMultiBackend
from utils.dataloaders import VID_FORMATS, LoadImages
from utils.general import (LOGGER, check_img_size, non_max_suppression, scale_coords, check_requirements, cv2,
                           increment_path, strip_optimizer, colorstr, print_args, check_file)
from utils.torch_utils import select_device, time_sync

from tracker.byte_tracker import BYTETracker
from tracking_utils.timer import Timer
from shapely.geometry import Point, Polygon, box

# remove duplicated stream handler to avoid duplicated logging
logging.getLogger().removeHandler(logging.getLogger().handlers[0])


def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)

    return color


def plot_tracking(image, tlwhs, obj_ids, id_mapping, labels=None, font='Arial.ttf', txt_color=(255, 255, 255)):
    # 画图的
    im = np.ascontiguousarray(np.copy(image))
    im = Image.fromarray(im)
    draw = ImageDraw.Draw(im)
    font = ImageFont.truetype(font, 18)

    line_thickness = 2

    for i, tlwh in enumerate(tlwhs):
        x1, y1, w, h = tlwh
        intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
        obj_id = id_mapping[int(obj_ids[i])]
        color = get_color(abs(obj_id))
        obj_id = obj_id if obj_id >= 0 else ""
        id_text = ''
        if labels is not None:
            id_text = '{}'.format(labels[i])
        if obj_id:
            id_text = id_text + ', {}'.format(obj_id)

        draw.rectangle(intbox, width=line_thickness, outline=color)  # box
        if id_text:
            w, h = font.getsize(id_text)  # text width, height
            outside = intbox[1] - h >= 0  # label fits outside box
            draw.rectangle(
                (intbox[0], intbox[1] - h if outside else intbox[1], intbox[0] + w + 1,
                 intbox[1] + 1 if outside else intbox[1] + h + 1),
                fill=color,
            )
            draw.text((intbox[0], intbox[1] - h if outside else intbox[1]), id_text, fill=txt_color, font=font)

        # cv2.rectangle(im, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness)
        # cv2.putText(im, id_text, (intbox[0], intbox[1]), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255),
        #             thickness=text_thickness)
    return np.asarray(im)


def filter_area(bbox: List, area_thres=0):
    #  检测框面积小于32^2的过滤掉
    if len(bbox) == 0:
        return bbox
    w = bbox[:, 2] - bbox[:, 0]
    h = bbox[:, 3] - bbox[:, 1]
    areas = torch.sqrt(w * h)
    mask = areas >= area_thres
    det = bbox[mask]
    res_det = bbox[~mask]
    return det, res_det


def linear_projection(area: float, area_boundary=(100, 480), count_boundary=(5, 15)):
    # 根据检测框的面积自适应调整id计数的threshold
    m = interp1d([area_boundary[1], area_boundary[0]], [count_boundary[0], count_boundary[1]])
    area = np.clip(area, area_boundary[0], area_boundary[1])
    return int(m(area))


def check_border(roi: Dict, img_shape=(1920, 1080)):
    # 检查roi是否在图像内部
    for roi_ in roi.values():
        roi_ = np.array(roi_)
        xmin, ymin = np.min(roi_, axis=0)
        xmax, ymax = np.max(roi_, axis=0)
        assert xmin >= 0 and ymin >= 0 and xmax <= img_shape[0] and ymax <= img_shape[1], \
            f'{roi_} out of image borader {img_shape}'


@torch.no_grad()
def run(
        source='0',
        yolo_weights=WEIGHTS / 'yolov5m.pt',  # model.pt path(s),
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.1,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        track_thresh=0.5,
        track_buffer=30,
        match_thresh=0.8,
        aspect_ratio_thresh=100,
        min_box_area=10,
        max_box_area=1000 * 1000,
        mot20=False,
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        show_vid=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        save_vid=False,  # save confidences in --save-txt labels
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/track',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        area_thres=0,
        count_thres=(15, 15),
        area_boundary=(100, 720),
):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    if not isinstance(yolo_weights, list):  # single yolo model
        exp_name = str(yolo_weights).rsplit('/', 1)[-1].split('.')[0]
    elif type(yolo_weights) is list and len(yolo_weights) == 1:  # single models after --yolo_weights
        exp_name = yolo_weights[0].split(".")[0]
    else:  # multiple models after --yolo_weights
        exp_name = 'ensemble'
    exp_name = name if name is not None else exp_name
    save_dir = increment_path(Path(project) / exp_name, exist_ok=exist_ok)  # increment run
    (save_dir / 'tracks' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    timer = Timer()
    # Load model
    device = select_device(device)
    model = DetectMultiBackend(yolo_weights, device=device, dnn=dnn, data=None, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
    nr_sources = 1
    vid_path, vid_writer, txt_path = [None] * nr_sources, [None] * nr_sources, [None] * nr_sources

    # Run tracking
    model.warmup(imgsz=(1 if pt else nr_sources, 3, *imgsz))  # warmup
    dt, seen = [0.0, 0.0, 0.0, 0.0], 0
    curr_frames, prev_frames = [None] * nr_sources, [None] * nr_sources

    #################################################################

    # init trackers, 4 class
    trackers = [BYTETracker(opt, frame_rate=8) for _ in range(4)]

    # roi = [[(200, 200), 800, 1080], [1120, 200, 1720, 1080]]
    # roi_base = dict(
    #     qianhai_mini={
    #         # 0: [(0, 0), (200, 0), (1920, 880), (1920, 1080), (1720, 1080), (0, 200)],
    #         # 0: [(0, 0), (1920, 780), (1920, 1080), (0, 300)],
    #         0: [[528, 726], [1408, 604], [1919, 792], [1919, 1079], [1510, 1079]],
    #     },
    #     qianhai_1_15fps={
    #         # 0: [(0, 0), (200, 0), (1920, 880), (1920, 1080), (1720, 1080), (0, 200)],
    #         # 0: [(0, 0), (1920, 780), (1920, 1080), (0, 300)],
    #         0: [[528, 726], [1408, 604], [1919, 792], [1919, 1079], [1510, 1079]],
    #     },
    #     HB05_15fps={
    #         0: [[260, 167], [897, 92], [1452, 723], [0, 924], [0, 253]]
    #     },
    #     SFJK_15fps={
    #         0: [[1401, 362], [731, 485], [579, 772], [1855, 452]]
    #     }
    # )
    roi_base = dict(
        qianhai_mini={
            # 0: [(0, 0), (200, 0), (1920, 880), (1920, 1080), (1720, 1080), (0, 200)],
            # 0: [(0, 0), (1920, 780), (1920, 1080), (0, 300)],
            0: [[844, 680], [1919, 1006], [1919, 1079], [697, 700]],
        },
        qianhai_1_15fps={
            # 0: [(0, 0), (200, 0), (1920, 880), (1920, 1080), (1720, 1080), (0, 200)],
            # 0: [(0, 0), (1920, 780), (1920, 1080), (0, 300)],
            0: [[844, 680], [1919, 1006], [1919, 1079], [697, 700]],
        },
        HB05_15fps={
            0: [[397, 380], [928, 307], [1215, 655], [835, 699], [41, 820]],
            1: [[196, 195], [494, 166], [494, 193], [129, 223]],
        },
        SFJK_15fps={
            0: [[711, 542], [1556, 409], [1789, 461], [635, 675]]
        }
    )

    _LABEL_MAP = {0: 'pedestrian', 1: 'car', 2: 'bus', 3: 'truck'}
    font = 'Arial.ttf'
    txt_color = (255, 255, 255)
    roi_points = roi_base[source.split('/')[-1]]
    check_border(roi_points)
    roi = dict()
    static_dict = dict()
    all_count = dict()
    output = dict()
    id_mapping = dict({-1: -1})
    id_count = 0
    # 初始化各种中间储存变量
    for roi_id, points in roi_points.items():
        roi[roi_id] = Polygon(points)
        static_dict.update({roi_id: {0: {}, 1: {}, 2: {}, 3: {}}})
        all_count.update({roi_id: [[], [], [], [], []]})
        output.update({roi_id: []})

    for frame_idx, (path, im, im0s, vid_cap, s) in tqdm(enumerate(dataset)):
        timer.tic()
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255.0  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path[0]).stem, mkdir=True) if opt.visualize else False
        pred = model(im, augment=opt.augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, opt.classes, opt.agnostic_nms,
                                   max_det=opt.max_det)
        dt[2] += time_sync() - t3

        # Process detections
        # batch size = 1
        for i, det in enumerate(pred):  # detections per image
            seen += 1

            p, im0, _ = path, im0s.copy(), getattr(dataset, 'frame', 0)
            p = Path(p)  # to Path
            # video file
            if source.endswith(VID_FORMATS):
                save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
            # folder with imgs
            else:
                save_path = str(save_dir / p.parent.name)  # im.jpg, vid.mp4, ...
            curr_frames[i] = im0

            s += '%gx%g ' % im.shape[2:]  # print string

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                det, res_det = filter_area(det, area_thres=area_thres)

                det = det.cpu().data.numpy()
                res_det = res_det.cpu().data.numpy()
                online_tlwhs = []
                online_cxcy = []
                online_labels = []
                online_ids = []
                for c in range(4):
                    x = det[(det[:, 5:6] == c).any(1)]
                    x_res = res_det[(res_det[:, 5:6] == c).any(1)]
                    if x_res is not None and len(x):
                        for x_ in x_res[:, :4]:
                            x_[2:] -= x_[:2]
                            online_tlwhs.append(x_)
                            online_ids.append(-1)
                            online_labels.append(_LABEL_MAP[c])
                    if x is not None and len(x):
                        online_targets = trackers[c].update(x[:, :5])
                        for t in online_targets:
                            tlwh = t.det_tlwh
                            xyxy = t.det_xyxy
                            cxcy = t.cxcy
                            tid = t.track_id
                            area = np.sqrt(tlwh[2] * tlwh[3])
                            if area <= area_thres:
                                tid = -1
                            online_tlwhs.append(tlwh)
                            online_cxcy.append(cxcy)
                            online_ids.append(tid)
                            online_labels.append(_LABEL_MAP[c])

                            if tid < 0:
                                continue
                            for roi_index, roi_ in roi.items():
                                # 检查目标框的中心点是否在roi内部
                                # id_in = roi_.contains(Point(cxcy))
                                if frame_idx == 0:
                                    id_in = False
                                else:
                                    id_in = roi_.intersection(box(*xyxy))

                            if tid not in static_dict[roi_index][c].keys():
                                static_dict[roi_index][c][tid] = [0, area, id_in]
                                if id_in:
                                    online_ids[-1] = -1
                            else:
                                if (not static_dict[roi_index][c][tid][-1]) and id_in:
                                    enter = True
                                else:
                                    enter = False

                                static_dict[roi_index][c][tid][-1] = id_in
                                    # 只有这条轨迹有一帧在roi内，就为True
                                static_dict[roi_index][c][tid][0] += 1
                                    # id的计数+1
                                static_dict[roi_index][c][tid][1] = (static_dict[roi_index][c][tid][1] *
                                                                         static_dict[roi_index][c][tid][0] + area) / (
                                                                                static_dict[roi_index][c][tid][0] + 1)
                                    # 滑动平均计算这条轨迹的面积
                                if static_dict[roi_index][c][tid][0] > 0 and tid not in all_count[roi_index][-1] and enter:
                                        # 如果这条轨迹出现在这个roi里面，且id的累计次数大于阈值，计数+1
                                    id_count += 1
                                    id_mapping[tid] = id_count
                                    all_count[roi_index][-1].append(tid)
                                    all_count[roi_index][c].append(tid)
                                ###############
                                if tid not in all_count[roi_index][-1]:
                                    online_ids[-1] = -1
                            if not id_in:
                                online_ids[-1] = -1
                timer.toc()
                online_im = plot_tracking(im0, online_tlwhs, online_ids, id_mapping, online_labels, font=font,
                                          txt_color=txt_color)
            else:
                timer.toc()
                online_im = im0

            # 下面是显示和画图的一些东西
            def get_postfix(c):
                s = ''
                for roi_index, _ in roi.items():
                    s += f'roi{roi_index}: {str(len(all_count[roi_index][c]))} '
                return s

            # for roi_ in roi_points.values():
            #     online_im = cv2.polylines(online_im, [np.array(roi_, dtype=np.int32)], 1, (0, 255, 0), thickness=2)

            # R=131 G=175 B=155
            mask = np.zeros(online_im.shape, dtype=np.int32)
            for roi_ in roi_points.values():
                mask = cv2.fillPoly(mask, [np.array(roi_, dtype=np.int32)], color=(255, 0, 0))

            online_im = (0.3 * mask + online_im).clip(0, 255).astype(np.uint8)

            online_im = Image.fromarray(online_im)
            draw = ImageDraw.Draw(online_im)
            pil_font = ImageFont.truetype(font, 40)

            draw.text((50, 50), f'pedestrian: {get_postfix(0)}', fill=(0, 0, 255), font=pil_font)
            draw.text((50, 100), f'car: {get_postfix(1)}', fill=(0, 0, 255), font=pil_font)
            draw.text((50, 150), f'bus: {get_postfix(2)}', fill=(0, 0, 255), font=pil_font)
            draw.text((50, 200), f'truck: {get_postfix(3)}', fill=(0, 0, 255), font=pil_font)
            draw.text((50, 250), f'all: {get_postfix(4)}', fill=(0, 0, 255), font=pil_font)
            online_im = np.asarray(online_im)

            # Save results (image with detections)
            if save_vid:
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
                    save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                    vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer[i].write(online_im)

            prev_frames[i] = curr_frames[i]

    for roi_index, _ in roi.items():
        output[roi_index] = [len(j) for j in all_count[roi_index]]
    print(id_mapping)

    #############################################################################
    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(
        f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms strong sort update per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_vid:
        s = f"\n{len(list(save_dir.glob('tracks/*.txt')))} tracks saved to {save_dir / 'tracks'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(yolo_weights)  # update model (to fix SourceChangeWarning)
    return output


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-weights', nargs='+', type=str, default=WEIGHTS / 'yolov5m.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='0', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action='store_true', help='display tracking video results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/track', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--area-thres', default=0, type=int)
    parser.add_argument('--count-thres', default=(15, 15), nargs=2, type=int)
    parser.add_argument('--area-boundary', default=(100, 720), nargs=2, type=int)
    parser.add_argument("--track-thresh", type=float, default=0.5, help="tracking confidence threshold")
    parser.add_argument("--track-buffer", type=int, default=10, help="the frames for keep lost tracks")
    parser.add_argument("--match-thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument(
        "--aspect_ratio_thresh", type=float, default=100,
        help="threshold for filtering out boxes of which aspect ratio are above the given value."
    )
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument('--max_box_area', type=float, default=1000 * 1000, help='filter out max boxes')
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")

    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(requirements=ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
    output = run(**vars(opt))
    print(output)


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
