import numpy as np
from PIL import Image, ImageDraw
import json
import cv2
import os
import glob
import argparse
import utils
from tqdm import tqdm


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

        if not os.path.isfile(json_path):
            mask_blank = np.array(
                Image.fromarray(np.zeros(
                    (560, 1476))).convert('L'))  #无标注返回黑的mask_blank
            Image.fromarray(mask_blank).save('data/pseudo_label/' +
                                             image_path.split('\\')[-1][:-4] +
                                             '.png')
            continue

        PED_index, SRF_IRF_index, mask_blank = utils.create_index(json_path)
        clsuters, neigbor_up, neigbor_all, label_slic, img = utils.create_SLIC_image(
            image_path, args.region_size, args.ruler)

        PED_mask, PED_short_mask = utils.get_PED_mask(PED_index, label_slic)
        SRF_IRF_mask = utils.get_SRF_IRF_mask(SRF_IRF_index, label_slic)

        truth_PED_mask = utils.get_PED_labels(PED_mask,
                                              neigbor_up,
                                              clsuters,
                                              img,
                                              Threshold=args.PED_threshold)
        truth_SRF_IRF_mask = utils.get_SRF_IRF_labels(
            SRF_IRF_mask,
            truth_PED_mask,
            neigbor_all,
            clsuters,
            img,
            Threshold=args.SRF_IRF_threshold)

        truth_PED_mask = utils.get_detach_PED(truth_PED_mask, neigbor_all, truth_SRF_IRF_mask)
        truth_mask = {**PED_short_mask, **truth_PED_mask, **truth_SRF_IRF_mask}
        truth_mask = utils.fill_holes(neigbor_all, truth_mask)
        for key, value in truth_mask.items():
            for positions in clsuters[key]:
                mask_blank[positions[0]][positions[1]] = value
        # Image.fromarray(mask_blank).show()
        mask = utils.mend(mask_blank)
        Image.fromarray(mask).save('data/pseudo_label/' +
                                   image_path.split('\\')[-1][:-4] + '.png')
