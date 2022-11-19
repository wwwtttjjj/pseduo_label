import numpy as np
from PIL import Image, ImageDraw
import json
import cv2
import os

labels2color = {"PED": 100, "SRF": 200, "IRF": 255}
'''usm sharping'''


def usm_edge_sharpening(img):
    blur_img = cv2.GaussianBlur(img, (0, 0), 5)
    usm = cv2.addWeighted(img, 1.5, blur_img, -0.5, 0)
    return usm


'''two point, get PED pixel index'''


def get_PED_index(start, end):
    points = []
    start_x = start[0]
    start_y = start[1]
    end_x = end[0]
    end_y = end[1]
    delta_x = end_x - start_x
    delta_y = end_y - start_y

    if abs(delta_x) > abs(delta_y):
        steps = abs(delta_x)
    else:
        steps = abs(delta_y)

    x_step = delta_x / steps
    y_step = delta_y / steps

    x = start_x
    y = start_y
    while steps >= 0:
        points.append([round(x), round(y)])
        x += x_step
        y += y_step
        steps -= 1
    return points


'''two type of data include points and line'''


def create_index(json_path):
    data = json.load(open(json_path, 'r', encoding='utf-8'))
    W, H = data['imageWidth'], data['imageHeight']
    # mask = Image.fromarray(np.zeros((H, W))).convert('L')
    mask_blank = np.array(Image.fromarray(np.zeros((H, W))).convert('L'))
    # mask1 = ImageDraw.Draw(mask)
    PED_index = []
    SRF_IRF_index = {}
    for points in data['shapes']:
        if points['label'] == 'PED':
            for [x, y] in points["points"]:
                PED_index.append([int(y), int(x)])
        else:
            [[x, y]] = points["points"]
            SRF_IRF_index[(int(y), int(x))] = labels2color[points['label']]
        # mask1.polygon(xy, fill = (labels2color[points['label']]))
    return PED_index, SRF_IRF_index, mask_blank


'''get the masked of PED'''


def get_PED_mask(PED_index, label_slic):
    PED_mask = {}
    PED_short_mask = {}

    for i in range(0, len(PED_index), 2):
        PED_xy = []
        start = PED_index[i]
        end = PED_index[i + 1]
        if start[1] > end[1]:
            start, end = end, start
        PED_xy += get_PED_index(PED_index[i], PED_index[i + 1])
        if len(PED_xy) < 3:
            for [x, y] in PED_xy:
                PED_short_mask[label_slic[x][y]] = labels2color['PED']
        else:
            for [x, y] in PED_xy:
                PED_mask[label_slic[x][y]] = labels2color['PED']
    return PED_mask, PED_short_mask


'''get the masked of SRF and IRF'''


def get_SRF_IRF_mask(SRF_IRF_index, label_slic):
    SRF_IRF_mask = {}
    for key, value in SRF_IRF_index.items():
        SRF_IRF_mask[label_slic[key[0]][key[1]]] = value
    return SRF_IRF_mask


'''generate the slic superpixel img'''


def create_SLIC_image(img_path, region_size=20, ruler=20, iterate=10):
    img = cv2.imread(img_path)
    # img = usm_edge_sharpening(img)
    #初始化slic项，超像素平均尺寸20（默认为10），平滑因子20
    slic = cv2.ximgproc.createSuperpixelSLIC(img,
                                             region_size=region_size,
                                             ruler=ruler)
    slic.iterate(iterate)  #迭代次数，越大效果越好
    mask_slic = slic.getLabelContourMask()  #获取Mask，超像素边缘Mask==1
    label_slic = slic.getLabels()  #获取超像素标签
    number_slic = slic.getNumberOfSuperpixels()  #获取超像素数目
    # mask_inv_slic = cv2.bitwise_not(mask_slic)
    # img_slic = cv2.bitwise_and(img,img,mask =  mask_inv_slic) #在原图上绘制超像素边界
    # cv2.imwrite('1.jpg',img_slic)
    W, H = label_slic.shape[:2]
    clsuters = [[] for _ in range(number_slic)]  # save the cluster
    neigbor_up = [[] for _ in range(number_slic)]  #save the neigbor relation
    for x in range(W):
        for y in range(H):
            clsuters[label_slic[x][y]].append([x, y])
            if mask_slic[x][y] == 255:
                for dx, dy in [(-1, 0)]:#(-1, -1), (-1, 1)
                    n_x, n_y = x + dx, y + dy
                    if n_x >= 0 and n_x < W and n_y >= 0 and n_y < H:
                        if label_slic[n_x][n_y] != label_slic[x][
                                y] and label_slic[n_x][n_y] not in neigbor_up[
                                    label_slic[x][y]] and len(
                                        neigbor_up[label_slic[x][y]]) < 2:
                            # print(1)
                            # break
                            neigbor_up[label_slic[x][y]].append(
                                label_slic[n_x][n_y])

    neigbor_all = [[] for _ in range(number_slic)]  #save the neigbor relation
    for x in range(W):
        for y in range(H):
            if mask_slic[x][y] == 255:
                for dx, dy in [(-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0),
                               (1, 1), (0, 1), (-1, 1)]:
                    n_x, n_y = x + dx, y + dy
                    if n_x >= 0 and n_x < W and n_y >= 0 and n_y < H:
                        if label_slic[n_x][n_y] != label_slic[x][
                                y] and label_slic[n_x][n_y] not in neigbor_all[
                                    label_slic[x][y]]:
                            # print(1)
                            # break
                            neigbor_all[label_slic[x][y]].append(
                                label_slic[n_x][n_y])
    return clsuters, neigbor_up, neigbor_all, label_slic, img


'''compute the dist matchscore (crop,mask)'''


def get_crop_mask(key, clsuters):
    minx, miny, maxx, maxy = float('inf'), float('inf'), 0, 0
    for [x, y] in clsuters[key]:
        if x < minx:
            minx = x
        if x > maxx:
            maxx = x
        if y < miny:
            miny = y
        if y > maxy:
            maxy = y
    mask = np.zeros((maxx - minx + 1, maxy - miny + 1), dtype=np.uint8)
    for [x, y] in clsuters[key]:
        mask[x - minx][y - miny] = 1
    return minx, miny, maxx, maxy, mask


'''compute the match_score (dist)'''


def get_hist_dice(key, current_key, clsuters, img):
    minx_c, miny_c, maxx_c, maxy_c, mask_c = get_crop_mask(
        current_key, clsuters)
    minx_k, miny_k, maxx_k, maxy_k, mask_k = get_crop_mask(key, clsuters)

    img_c = img[minx_c:maxx_c + 1, miny_c:maxy_c + 1]
    img_k = img[minx_k:maxx_k + 1, miny_k:maxy_k + 1]
    # img_c = img.crop((miny_c, minx_c, maxy_c + 1, maxx_c + 1))
    # img_k = img.crop((miny_k, minx_k, maxy_k + 1, maxx_k + 1))

    # img_c =  cv2.cvtColor(np.asarray(img_c),cv2.COLOR_RGB2BGR)
    # img_k = cv2.cvtColor(np.asarray(img_k),cv2.COLOR_RGB2BGR)

    img_c = cv2.cvtColor(img_c, cv2.COLOR_BGR2GRAY)
    img_k = cv2.cvtColor(img_k, cv2.COLOR_BGR2GRAY)

    H_c = cv2.calcHist([img_c], [0], mask_c, [256], [0, 256])
    H_c = cv2.normalize(H_c, H_c, 0, 1, cv2.NORM_MINMAX, -1)

    H_k = cv2.calcHist([img_k], [0], mask_k, [256], [0, 256])
    H_k = cv2.normalize(H_k, H_k, 0, 1, cv2.NORM_MINMAX, -1)
    match_score = cv2.compareHist(H_c, H_k, method=cv2.HISTCMP_CORREL)
    return match_score


'''cos dice'''


def cosine_similarity(x, y):
    num = x.dot(y.T)
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    return num / denom


def get_cosin_dice(key, current_key, clsuters, img):
    img_gary = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    frequent_key = np.zeros(256)
    frequent_cur_key = np.zeros(256)
    for [x, y] in clsuters[key]:
        frequent_key[img_gary[x][y]] += 1
    for [x, y] in clsuters[current_key]:
        frequent_cur_key[img_gary[x][y]] += 1

    return cosine_similarity(frequent_cur_key, frequent_key)


'''Expand the field of pixel blocks corresponding to weak labels(PED)'''


def get_PED_labels(masked_index, neigbor, clsuters, img, Threshold=0.5):
    if len(masked_index) < 3:
        return masked_index
    stack = []
    already = []
    truth_mask = {}
    # add_t = 0.001
    for key, value in masked_index.items():
        already.append(key)
        stack.append(key)
        truth_mask[key] = value
        num = 0
        while stack:
            current_key = stack.pop(0)
            dice = get_cosin_dice(key, current_key, clsuters, img)
            # print(key, current_key, dice)
            if dice >= Threshold:
                neigbor_keys = neigbor[current_key]
                # print(neigbor_keys)
                for neigbor_key in neigbor_keys:
                    if neigbor_key not in already:
                        already.append(neigbor_key)
                        stack.append(neigbor_key)
                truth_mask[current_key] = value
                # Threshold += add_t
                num += 1
            if num >= 9:
                break
            else:
                continue
    return truth_mask


'''Expand the field of pixel blocks corresponding to weak labels(SRF and IRF)'''


def get_SRF_IRF_labels(masked_index,
                       truth_PED_mask,
                       neigbor,
                       clsuters,
                       img,
                       Threshold=0.5):
    stack = []
    already = []
    truth_mask = {}
    # add_t = 0.001
    for key, value in masked_index.items():
        already.append(key)
        stack.append(key)
        truth_mask[key] = value
        threshold = Threshold
        while stack:
            current_key = stack.pop(0)
            dice = get_cosin_dice(key, current_key, clsuters, img)
            # print(key, current_key, dice)
            if dice >= threshold:
                neigbor_keys = neigbor[current_key]
                # print(neigbor_keys)
                for neigbor_key in neigbor_keys:
                    if neigbor_key not in already and neigbor_key not in truth_PED_mask:
                        already.append(neigbor_key)
                        stack.append(neigbor_key)
                truth_mask[current_key] = value
                # threshold += add_t
            else:
                continue
    return truth_mask


'''fill the holes'''


def fill_holes(neigbor, truth_mask):
    probabilty = []
    add_mask = {}

    # truth_mask_fill_hole = truth_mask
    for key, value in truth_mask.items():
        for n in neigbor[key]:
            # if n not in truth_mask:
            probabilty.append(n)
    for p in probabilty:
        l = len(neigbor[p])
        num = 0
        for n in neigbor[p]:
            if n in truth_mask and num == 0:
                value = truth_mask[n]
                num += 1
            elif n in truth_mask and num != 0:
                if truth_mask[n] == value:
                    num += 1
                else:
                    break
            else:
                break
        if  num >= l - 2 :
            add_mask[p] = value
    return {**truth_mask, **add_mask}


# mend some pixels
def mend(mask):
    return cv2.medianBlur(mask, 5)
#delete some nei PED
def get_detach_PED(truth_PED_mask, neigbor, truth_SRF_IRF_mask):
    del_key = []

    # truth_mask_fill_hole = truth_mask
    for key, value in truth_PED_mask.items():
        for n in neigbor[key]:
            if n in truth_SRF_IRF_mask:
                del_key.append(key)
                continue
    for k in del_key:
        truth_PED_mask.pop(k, None)
    return truth_PED_mask


    