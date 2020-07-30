import os
import cv2
from PIL import Image
import argparse

def jpg2video(imgs_dir, fps):
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    images = os.listdir(imgs_dir)
    video = imgs_dir + '.avi'
    im = Image.open(os.path.join(imgs_dir, images[0]))
    vw = cv2.VideoWriter(video, fourcc, fps, im.size)

    os.chdir(imgs_dir)
    for image in range(len(images)):
        # Image.open(str(image)+'.jpg').convert("RGB").save(str(image)+'.jpg')
        jpgfile = '%04d.jpg' % (image+1)
        print(jpgfile)
        try:
            frame = cv2.imread(jpgfile)
            vw.write(frame)
        except Exception as exc:
            print(jpgfile, exc)
    vw.release()
    print(video, 'Synthetic success!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Images to video.")
    parser.add_argument('--imgs_dir', default=None, type=str)
    parser.add_argument('--fps', default=24, type=int)
    args = parser.parse_args()
    jpg2video(args.imgs_dir, args.fps)