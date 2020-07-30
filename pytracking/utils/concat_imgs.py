import os
import numpy as np
import cv2
import argparse
from PIL import ImageFont, ImageDraw, Image

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Images to video.")
    parser.add_argument('--imgs1_dir', default=None, type=str)
    parser.add_argument('--imgs2_dir', default=None, type=str)
    parser.add_argument('--save_dir', default=None, type=str)
    args = parser.parse_args()

    img_list1 = os.listdir(args.imgs1_dir)
    img_list2 = os.listdir(args.imgs2_dir)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    num_imgs = len(img_list1)

    for i in range(num_imgs):
        img_name = '%04d.jpg' % (i + 1)
        path_1 = os.path.join(args.imgs1_dir, img_name)
        path_2 = os.path.join(args.imgs2_dir, img_name)
        save_path = os.path.join(args.save_dir, img_name)

        im_1 = cv2.imread(path_1, 1)  # python2: cv2.CV_LOAD_IMAGE_COLOR; python3: cv2.IMREAD_COLOR
        im_1 = cv2.copyMakeBorder(im_1, 50, 0, 0, 0, cv2.BORDER_CONSTANT, value=0)

        im_1 = cv2.putText(im_1, "Results of FCOT", (20, 30), cv2.FONT_HERSHEY_DUPLEX,
                    0.6, (255, 255, 255), 1)

        im_2 = cv2.imread(path_2, 1)  # python2: cv2.CV_LOAD_IMAGE_COLOR; python3: cv2.IMREAD_COLOR
        im_2 = cv2.copyMakeBorder(im_2, 50, 0, 0, 0, cv2.BORDER_CONSTANT, value=0)

        im_2 = cv2.putText(im_2, "Results of DiMP", (20, 30), cv2.FONT_HERSHEY_DUPLEX,
                    0.6, (255, 255, 255), 1)

        im_merge = np.concatenate([im_1, im_2], 1)

        cv2.imwrite(save_path, im_merge)

