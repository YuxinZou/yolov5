dir=./images/testset_images

for file in $dir/*;do
    echo $file
    ~/miniconda3/envs/yolov5/bin/python track_bytetrack.py --yolo-weights ../tmp/yolov5/weights/fusion2/best.pt --source $file --imgsz 736 1280 --name track --exist-ok --save-vid --conf-thres 0.4 --line-thickness 1
done
