import numpy as np
from PIL import Image, ImageDraw
import json
import cv2
import os
import glob
import argparse
import utils
from tqdm import tqdm
import math


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--PED_threshold',
                        type=float,
                        default=0.65,
                        help='the threshold of hist compare distance [0, 1]')
    parser.add_argument('--SRF_IRF_threshold',
                        type=float,
                        default=0.75,
                        help='the threshold of hist compare distance [0, 1]')

    parser.add_argument('--region_size',
                        type=int,
                        default=20,
                        help='the region_size of superpixel')
    parser.add_argument('--ruler',
                        type=int,
                        default=20,
                        help='the ruler of superpixel')
    parser.add_argument('--data_path',
                        type=str,
                        default='data\\test_data',
                        help='the datasets of images and jsons')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    '''path'''
    jpg_paths = []
    json_paths = []

    for file in glob.glob(os.path.join(args.data_path, '*.jpg')):

        jpg_paths.append(file)
        json_paths.append(file[:-3] + 'json')

    for i in tqdm(range(len(jpg_paths))):

        image_path = jpg_paths[i]
        json_path = json_paths[i]

        img = cv2.imread(image_path)
        W, H = img.shape[0], img.shape[1]
        probability_map = np.array(Image.fromarray(np.ones(
            (W, H))).convert('L'),dtype=float)

        #全阴性
        if not os.path.isfile(json_path):
            mask_blank = np.array(
                Image.fromarray(np.zeros(
                    (W, H))).convert('L'))  #无标注返回全0的mask_blank
            Image.fromarray(mask_blank).save('data/pseudo_label/' +
                                             image_path.split('\\')[-1][:-4] +
                                             '.png')
            np.save(
                'data/probability_map/' + image_path.split('\\')[-1][:-4] +
                '.npy', probability_map)

            continue

        PED_index, SRF_IRF_index, mask_blank = utils.create_index(json_path)
        clsuters, neigbor_up, neigbor_all, label_slic, img, xy_center = utils.create_SLIC_image(
            image_path, args.region_size, args.ruler)
        proximity_distance = (1.25) * args.region_size * math.sqrt(2)  #标准一格距离度量
        PED_mask, PED_short_mask = utils.get_PED_mask(PED_index, label_slic)
        SRF_IRF_mask = utils.get_SRF_IRF_mask(SRF_IRF_index, label_slic)

        truth_PED_mask, probability_map = utils.get_PED_labels(
            PED_mask,
            neigbor_up,
            clsuters,
            img,
            xy_center,
            probability_map,
            proximity_distance,
            Threshold=args.PED_threshold,
        )
        truth_SRF_IRF_mask, probability_map = utils.get_SRF_IRF_labels(
            SRF_IRF_mask,
            truth_PED_mask,
            neigbor_all,
            clsuters,
            img,
            xy_center,
            probability_map,
            proximity_distance,
            Threshold=args.SRF_IRF_threshold)

        truth_PED_mask = utils.get_detach_PED(truth_PED_mask, neigbor_all,
                                              truth_SRF_IRF_mask)
        truth_mask = {**PED_short_mask, **truth_PED_mask, **truth_SRF_IRF_mask}
        truth_mask, probability_map = utils.fill_holes(clsuters, neigbor_all, truth_mask, probability_map)
        for key, value in truth_mask.items():
            for positions in clsuters[key]:
                mask_blank[positions[0]][positions[1]] = value
        # Image.fromarray(mask_blank).show()
        mask, probability_map = utils.mend(mask_blank, probability_map)

        Image.fromarray(mask).save('data/pseudo_label/' +
                                   image_path.split('\\')[-1][:-4] + '.png')

        np.save(
            'data/probability_map/' + image_path.split('\\')[-1][:-4] + '.npy',
            probability_map)
