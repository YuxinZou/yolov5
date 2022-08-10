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
import torch.backends.cudnn as cudnn
from scipy.interpolate import interp1d
from tqdm import tqdm

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
from utils.dataloaders import VID_FORMATS, LoadImages, LoadStreams, LoadImagesSetFps
from utils.general import (LOGGER, check_img_size, non_max_suppression, scale_coords, check_requirements, cv2,
                           check_imshow, xyxy2xywh, increment_path, strip_optimizer, colorstr, print_args, check_file)
from utils.torch_utils import select_device, time_sync
from utils.plots import Annotator, colors, save_one_box

from tracker.byte_tracker import BYTETracker
from tracking_utils.timer import Timer
from utils.visualize import plot_tracking
from loguru import logger

# remove duplicated stream handler to avoid duplicated logging
logging.getLogger().removeHandler(logging.getLogger().handlers[0])


def filter_area(bbox, area_thres=0):
    if len(bbox) == 0:
        return bbox
    w = bbox[:, 2] - bbox[:, 0]
    h = bbox[:, 3] - bbox[:, 1]
    areas = torch.sqrt(w * h)
    mask = areas >= area_thres
    det = bbox[mask]
    return det


def linear_projection(area, area_boundary=(100, 720), count_boundary=(15, 15)):
    # 100 -> 720
    m = interp1d([area_boundary[1], area_boundary[0]], [count_boundary[0], count_boundary[1]])
    area = np.clip(area, area_boundary[0], area_boundary[1])
    return int(m(area))


def check_border(roi, img_shape=(1920, 1080)):
    for roi_ in roi:
        assert roi_[0] >= 0 and roi_[1] >= 0 and roi_[2] <= img_shape[0] and roi_[3] <= img_shape[1], \
            f'{roi_} out of image borader {img_shape}'
    print('check passed')


def check_in_roi(coord, roi):
    if (roi[0] < coord[0] < roi[2]) and (roi[1] < coord[1] < roi[3]):
        return True
    else:
        return False


@torch.no_grad()
def run(
        source='0',
        yolo_weights=WEIGHTS / 'yolov5m.pt',  # model.pt path(s),
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
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
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        hide_class=False,  # hide IDs
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        area_thres=0,
        interval=3,
        count_thres=(15, 15),
        area_boundary=(100, 720),
        gt='gt.txt',
        use_det=False,
        roi=(200, 200, 1720, 1080),
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
    trackers = [BYTETracker(opt, frame_rate=8) for _ in range(4)]
    timer = Timer()
    # Load model
    device = select_device(device)
    model = DetectMultiBackend(yolo_weights, device=device, dnn=dnn, data=None, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    if webcam:
        show_vid = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        nr_sources = len(dataset)
    else:
        # dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        dataset = LoadImagesSetFps(source, img_size=imgsz, stride=stride, auto=pt, fps=interval)
        nr_sources = 1
    vid_path, vid_writer, txt_path = [None] * nr_sources, [None] * nr_sources, [None] * nr_sources

    outputs = [None] * nr_sources

    # Run tracking
    model.warmup(imgsz=(1 if pt else nr_sources, 3, *imgsz))  # warmup
    dt, seen = [0.0, 0.0, 0.0, 0.0], 0
    curr_frames, prev_frames = [None] * nr_sources, [None] * nr_sources

    roi = np.array(roi)
    if len(roi.shape) == 1:
        roi = roi[np.newaxis, :]
    check_border(roi)

    static_dict = dict()
    all_count = dict()
    for i in range(len(roi)):
        static_dict.update({i: {0: {}, 1: {}, 2: {}, 3: {}}})
        all_count.update({i: [[], [], [], [], []]})

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
                txt_file_name = p.stem
                save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
            # folder with imgs
            else:
                txt_file_name = p.parent.name  # get folder name containing current img
                save_path = str(save_dir / p.parent.name)  # im.jpg, vid.mp4, ...
            curr_frames[i] = im0

            txt_path = str(save_dir / 'tracks' / txt_file_name)  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            imc = im0.copy() if save_crop else im0  # for save_crop

            annotator = Annotator(im0, line_width=2, pil=not ascii)

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size

                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                det = filter_area(det, area_thres=area_thres)

                det = det.cpu().data.numpy()
                online_tlwhs = []
                online_cxcy = []
                online_ids = []
                online_scores = []
                for c in range(4):
                    x = det[(det[:, 5:6] == c).any(1)]
                    if x is not None and len(x):
                        online_targets = trackers[c].update(x[:, :5])
                        for t in online_targets:
                            if use_det:
                                tlwh = t.det_tlwh
                            else:
                                tlwh = t.tlwh
                            cxcy = t.cxcy
                            tid = t.track_id
                            online_tlwhs.append(tlwh)
                            online_cxcy.append(cxcy)
                            area = np.sqrt(tlwh[2] * tlwh[3])
                            online_ids.append(tid)
                            online_scores.append(t.score)

                            for roi_index, roi_ in enumerate(roi):

                                id_in = check_in_roi(cxcy, roi_)

                                if tid not in static_dict[roi_index][c].keys():
                                    static_dict[roi_index][c][tid] = [0, area, id_in]
                                else:
                                    static_dict[roi_index][c][tid][-1] = static_dict[roi_index][c][tid][-1] or id_in
                                    static_dict[roi_index][c][tid][0] += 1
                                    static_dict[roi_index][c][tid][1] = (static_dict[roi_index][c][tid][1] *
                                                                         static_dict[roi_index][c][tid][0] + area) / (
                                                                                static_dict[roi_index][c][tid][0] + 1)
                                    if static_dict[roi_index][c][tid][0] > linear_projection(
                                            static_dict[roi_index][c][tid][1], area_boundary,
                                            count_thres) and tid not in all_count[roi_index][-1] and static_dict[roi_index][c][tid][-1]:
                                        all_count[roi_index][-1].append(tid)
                                        all_count[roi_index][c].append(tid)

                timer.toc()
                online_im = plot_tracking(
                    im0, online_tlwhs, online_ids, frame_id=frame_idx + 1, fps=1. / timer.average_time, online_cxcy=online_cxcy)
            else:
                timer.toc()
                online_im = im0

            def get_postfix(c):
                return str(all_count[0][c][-1]) if len(all_count[0][c]) else ''

            online_im = cv2.rectangle(online_im, (roi[0][0], roi[0][1]), (roi[0][2], roi[0][3]), (0, 255, 0), 2)
            online_im = cv2.putText(online_im, 'all: ' + str(len(all_count[0][-1])), (50, 300), cv2.FONT_HERSHEY_SIMPLEX,
                                    1, (255, 0, 0), 2, cv2.LINE_AA)
            online_im = cv2.putText(online_im, f'people: {str(len(all_count[0][0]))} {str(get_postfix(0))}', (50, 100),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1, (255, 0, 0), 2, cv2.LINE_AA)
            online_im = cv2.putText(online_im, f'car: {str(len(all_count[0][1]))} {str(get_postfix(1))}', (50, 150),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (255, 0, 0), 2, cv2.LINE_AA)
            online_im = cv2.putText(online_im, f'bus: {str(len(all_count[0][2]))} {str(get_postfix(2))}', (50, 200),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (255, 0, 0), 2, cv2.LINE_AA)
            online_im = cv2.putText(online_im, f'truck: {str(len(all_count[0][3]))} {str(get_postfix(3))}', (50, 250),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1, (255, 0, 0), 2, cv2.LINE_AA)

            # Stream results
            im0 = annotator.result()
            if show_vid:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

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

    print([len(i) for i in all_count[0]])
    txt_path = str(save_dir / 'result.txt')
    with open(txt_path, 'a+') as f:
        s_ = ' '.join(map(str, [len(i) for i in all_count[0]]))
        s_ = source + ' ' + s_
        print(s_)
        f.write(s_ + '\n')
    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(
        f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms strong sort update per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_vid:
        s = f"\n{len(list(save_dir.glob('tracks/*.txt')))} tracks saved to {save_dir / 'tracks'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(yolo_weights)  # update model (to fix SourceChangeWarning)
    return s_, static_dict


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
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--hide-class', default=False, action='store_true', help='hide IDs')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--area-thres', default=0, type=int)
    parser.add_argument('--interval', default=3, type=int)
    parser.add_argument('--count-thres', default=(15, 15), nargs=2, type=int)
    parser.add_argument('--area-boundary', default=(100, 720), nargs=2, type=int)
    parser.add_argument('--roi', default=(200, 200, 1720, 1080), nargs=4, type=int)
    parser.add_argument('--use-det', default=False, action='store_true', help='hide IDs')
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
    parser.add_argument('--gt', default='gt.txt', help='gt file')

    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def parse(txt_file):
    res = {}
    data = open(txt_file, 'r', encoding='UTF-8').readlines()
    for d in data:
        d = d.strip().split(' ')
        name = d[0].split('/')[-1].split('.')[0]
        res[name] = list(map(int, d[1:]))
    return res


def metric(data1, data2):
    res = {}
    classes = (0, 1, 2, 3)
    for c in classes:
        tp, fp, fn = 0, 0, 0
        for key, gt in data1.items():
            dt = data2[key]
            g = gt[c]
            d = dt[c]
            tp += min(g, d)
            fp += max(0, d - g)
            fn += max(0, g - d)
        res[c] = dict(tp=tp, fp=fp, fn=fn, precision=tp / (tp + fp), recall=tp / (tp + fn), acc=tp / (tp + fp + fn))
    return res


def main(opt):
    check_requirements(requirements=ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
    source = opt.source
    if source.endswith('.mp4'):
        print(opt)
        run(**vars(opt))
        # import pickle
        # fw = open('./json/result.pkl', 'wb')
        # pickle.dump(dict(id_dict=josn_res, ratio_dict=ratio_dict), fw)
    else:
        dt = {}
        dist_dict = {}
        for fname in sorted(os.listdir(source)):
            opt.source = os.path.join(source, fname)
            s, josn_res, ratio_dict = run(**vars(opt))
            s = s.strip().split(' ')
            name = s[0].split('/')[-1].split('.')[0]
            dt[name] = list(map(int, s[1:]))
            dist_dict[name] = dict(id_dict=josn_res, ratio_dict=ratio_dict)
        import pickle
        fw = open('./json/result.pkl', 'wb')
        pickle.dump(dist_dict, fw)
        gt = parse(f'./txt/{opt.gt}')
        res = metric(gt, dt)
        for k, v in res.items():
            print(k)
            print(v)
        print(np.mean([v['acc'] for k, v in res.items()]))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
