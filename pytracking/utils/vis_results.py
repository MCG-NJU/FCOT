import cv2
import argparse
import os
import numpy as np

# ours color: (0, 0, 255)  gt color: (0, 255, 0)
def draw_box(img, bbox, color, save_dir=None):
    if color == 'red':
        color = (0, 0, 255)
    elif color == 'green':
        color = (0, 255, 0)
    elif color == 'yellow':
        color = (0, 255, 255)
    else:
        color = (255, 0, 0)

    # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    draw_rec = cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])), color, 2)
    # draw_rec = cv2.rectangle(img, (bbox[1], bbox[0]), (bbox[3], bbox[2]), color, 2)
    cv2.imwrite(save_dir, draw_rec)


"""
usage:
(1) ours
python vis_results.py --imgs_dir ~/data/OTB100/Liquor/img --save_dir ~/data/OTB100_results/Liquor_ours \
--box_results ~/data/OTB100_results/OnlineClsAndReg_otb_results/Liquor.txt --color red
    
(2) dimp
python vis_results.py --imgs_dir ~/data/OTB100/Liquor/img --save_dir ~/data/OTB100_results/Liquor_dimp \
--box_results ~/data/OTB100_results/dimp50/Liquor.txt --color blue
    
(3) gt
python vis_results.py --imgs_dir ~/data/OTB100/Liquor/img --save_dir ~/data/OTB100_results/Liquor_gt \
--box_results ~/data/OTB100/Liquor/groundtruth_rect.txt --color green
    
(4) merge ours and gt 
python vis_results.py --imgs_dir ~/data/OTB100_results/Liquor_gt --save_dir ~/data/OTB100_results/Liquor_oursGT \
--box_results ~/data/OTB100_results/OnlineClsAndReg_otb_results/Liquor.txt --color red
    
(5) merge dimp and gt
python vis_results.py --imgs_dir ~/data/OTB100_results/Liquor_gt --save_dir ~/data/OTB100_results/Liquor_dimpGT \
--box_results ~/data/OTB100_results/dimp50/Liquor.txt --color blue
"""
def main():
    parser = argparse.ArgumentParser(description="Draw box")
    parser.add_argument('--imgs_dir', default=None, type=str)
    parser.add_argument('--save_dir', default=None, type=str)
    parser.add_argument('--box_results', default=None, type=str)
    parser.add_argument('--color', default='red', type=str)
    args = parser.parse_args()

    imgs_dir = args.imgs_dir
    box_results = args.box_results
    imgs = os.listdir(imgs_dir)
    save_dir = args.save_dir

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    try:
        box_results = np.loadtxt(str(box_results), dtype=np.float64)
    except:
        box_results = np.loadtxt(str(box_results), delimiter=',', dtype=np.float64)

    for i in range(len(imgs)):
        img = imgs[i]
        print(img)
        img_num = int(img.split('.')[0])
        # if img_num > 74:
        #     continue
        # print(img_num)
        box_result = box_results[img_num-1, :]
        img_dir = os.path.join(imgs_dir, img)
        image = cv2.imread(img_dir)
        img_save_dir = os.path.join(save_dir, img)
        print(box_result)
        draw_box(image, bbox=box_result, color=args.color, save_dir=img_save_dir)


if __name__ == "__main__":
    main()