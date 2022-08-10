import subprocess
import os


def split_video():
    path = 'D:/data/前海/test0628'
    dst = path # 'D:/data/前海/新增28路数据_流量测试集/'
    os.makedirs(dst, exist_ok=True)
    for (root, dirs, files) in os.walk(path):
        for file in files:
            print(file)
            video_name = f'{root}/{file}'
            video_dst_name = f'{dst}/{file}'
            video_dst_name = video_dst_name.replace(".mp4", "_cut.mp4")
            cmd = f'ffmpeg -ss 00:00:00 -t 00:05:00 -accurate_seek -i {video_name} -codec copy -avoid_negative_ts 1 {video_dst_name}'
            subprocess.call(cmd, shell=True)

def split_image():
    path = 'D:/data/前海/新增28路数据/'

    for (root, dirs, files) in os.walk(path):
        for file in files:
            video_name = f'{root}/{file}'
            dst = video_name.split('.')[0]
            dst = dst.replace('新增28路数据', 'new_16')
            os.makedirs(dst, exist_ok=True)
            cmd = f'ffmpeg -ss 00:05:00 -i {video_name} -vf fps=fps=1/4 -q:v 2 -f image2 {dst}/image_%08d.jpg'
            subprocess.call(cmd, shell=True)



if __name__ == '__main__':
    split_video()