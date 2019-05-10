import os
import cv2
import numpy as np
import pickle
import math
from shutil import copyfile
from tqdm import trange, tqdm
from PIL import Image
import argparse


def obtain_ann_dict(img_root, ann_root):
    ann_dict = {}
    assert (os.path.isdir(img_root)), 'Path does not exist: {}'.format(img_root)
    assert (os.path.isdir(ann_root)), 'Path does not exist: {}'.format(ann_root)

    for add in os.listdir(img_root):
        ann_dict[add] = []

    for add in os.listdir(ann_root):
        ann_dict[add[:-10] + '.jpg'].append(add)
    return ann_dict


def get_data(data_root, split_name, mode, num_split, num_idx):
    assert (split_name in ['train', 'val'])

    set_list_add = split_name + '.txt'

    list_root = os.path.join(data_root, 'list')
    img_root = os.path.join(data_root, split_name, 'images')
    ann_root = os.path.join(data_root, split_name, 'parsing_annos')  # '/annotations/'

    ann_dict = obtain_ann_dict(img_root, ann_root)

    flist = [line.strip() for line in open(os.path.join(list_root, set_list_add)).readlines()]
    num_samples = len(flist)
    group_size = int(math.ceil(num_samples * 1.0 / num_split))
    start_idx = num_idx * group_size
    end_idx = min((num_idx + 1) * group_size, num_samples)
    flist = flist[start_idx:end_idx] 

    list_dat = []
    for add in tqdm(flist, desc='Loading %s ..' % (set_list_add)):
        dat = {}
        im_path = os.path.join(img_root, add + '.jpg')
        im_sz = cv2.imread(im_path).shape
        dat['filename'] = add
        dat['filepath'] = im_path
        if mode == 'mask':
            dat['width'] = im_sz[1]
            dat['height'] = im_sz[0]
            dat['bboxes'] = []
            for ann_add in sorted(ann_dict[add + '.jpg']):
                ann = np.array(Image.open(os.path.join(ann_root, ann_add)))
                if len(ann.shape) == 3:
                    ann = ann[:, :, 0]  # Make sure ann is a two dimensional np array.
                ys, xs = np.where(ann > 0)
                x1, x2, y1, y2 = xs.min(), xs.max(), ys.min(), ys.max()
                dat['bboxes'].append(
                    {'class': 'person',
                     'ann_path': os.path.join(ann_root, ann_add),
                     'x1': x1,
                     'y1': y1,
                     'x2': x2,
                     'y2': y2})

        list_dat.append(dat)

    return list_dat


def show_data(list_dat, num=4):
    from pylab import plt
    for dat in np.random.choice(list_dat, num):
        print(dat)
        im = cv2.imread(dat['filepath'])[:, :, ::-1]
        plt.figure(1)
        plt.imshow(im)
        for bbox in dat['bboxes']:
            plt.gca().add_patch(plt.Rectangle((bbox['x1'], bbox['y1']),
                                              bbox['x2'] - bbox['x1'],
                                              bbox['y2'] - bbox['y1'], fill=False,
                                              edgecolor='red', linewidth=1))
        for idx, bbox in enumerate(dat['bboxes']):
            ann = np.array(Image.open(bbox['ann_path']))
            if len(ann.shape) == 3:
                ann = ann[:, :, 0]  # Make sure ann is a two dimensional np array.
            plt.figure(11 + idx)
            plt.imshow(ann)
        plt.show()

def dfs(mask, new_mask, h_idx, w_idx, label, flag):
    coord_stack = []
    coord_stack.append([h_idx, w_idx])
    count = 0
    while len(coord_stack) != 0:
        coord = coord_stack.pop()
        h_idx = coord[0]
        w_idx = coord[1]
        if h_idx < 0 or h_idx >= mask.shape[0] or w_idx < 0 or w_idx >= mask.shape[1]:
            continue
        if new_mask[h_idx][w_idx] != -1 or mask[h_idx][w_idx] != label:
            continue
        coord_stack.extend([[h_idx - 1, w_idx - 1],
                            [h_idx - 1, w_idx],
                            [h_idx - 1, w_idx + 1],
                            [h_idx, w_idx - 1],
                            [h_idx, w_idx + 1],
                            [h_idx + 1, w_idx - 1],
                            [h_idx + 1, w_idx],
                            [h_idx + 1, w_idx + 1]])
        new_mask[h_idx][w_idx] = flag
        count += 1
    
    return count


def polish_mask(mask):
    height = mask.shape[0]
    width = mask.shape[1]
    # flag: count
    connected_comp = []

    # flag
    new_mask = np.full((height, width), -1)
    flag = 0
    for i in range(height):
        for j in range(width):
            if new_mask[i][j] == -1:  # not visited
                count = dfs(mask, new_mask, i, j, mask[i][j], flag)
                connected_comp.append(count)
                flag += 1
    limited_size = int((height * width) * 0.001)
    for i in range(height):
        for j in range(width):
            flag = new_mask[i][j]
            if connected_comp[flag] < limited_size:
                mask[i][j] = -1 * (mask[i][j] - 1) # flip: 0->1, 1->0
    return mask

def transform_mask(list_data):
    file_prefix = 'mhp_'
    output_dir = 'output'
    out_mask_dir = os.path.join(output_dir, 'masks')
    if not os.path.exists(out_mask_dir):
        os.makedirs(out_mask_dir)
    for dat in list_data:
        print('convert', dat['filename'])
        width = dat['width']
        height = dat['height']
        mask = np.zeros((height, width), dtype=np.float32)
        for idx, anno in enumerate(dat['bboxes']):
            m = np.array(Image.open(anno['ann_path']), dtype=np.float32)
            if len(m.shape) == 3:
                m = m[:, :, 0]  # Make sure ann is a two dimensional np array.
            if m.shape[0] != height or m.shape[1] != width:
                print('annotations not match for %s' % dat['filepath'])
                mask = None
                break
            mask[m > 0] = 1
        new_file_name = file_prefix + dat['filename']
        mask = mask.astype(dtype=np.uint8)
        mask = polish_mask(mask).astype(dtype=np.uint8)
        mask_img = Image.fromarray(mask)
        out_mask_path = os.path.join(out_mask_dir, new_file_name + '.png')
        with open(out_mask_path, 'w') as f:
            mask_img.save(out_mask_path, 'PNG')

def transform_img(list_data, split_name):
    file_prefix = 'mhp_'
    output_dir = 'output'
    out_img_dir = os.path.join(output_dir, 'images')
    out_list_dir = os.path.join(output_dir, 'segmentation')
    if not os.path.exists(out_img_dir):
        os.makedirs(out_img_dir)
    if not os.path.exists(out_list_dir):
        os.makedirs(out_list_dir)
    for dat in list_data:
        new_file_name = file_prefix + dat['filename']
        out_img_path = os.path.join(out_img_dir, new_file_name + '.jpg')
        copyfile(dat['filepath'], out_img_path)
        # write list file
        list_file_path = os.path.join(out_list_dir, split_name + '.txt')
        with open(list_file_path, "a") as f:
            f.write(new_file_name + '\n')
    print('total: %d' % (len(list_data)))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default='LV-MHP-v2',
                        help="dataset directory")
    parser.add_argument("--split_name", type=str, default='val',
                        help="split name[train|val]")
    parser.add_argument("--mode", type=str, default='image',
                        help="mode[image|mask]")
    parser.add_argument("--num_split", type=int, default=10,
                        help="")
    parser.add_argument("--num_idx", type=int, default=0,
                        help="")
    return parser.parse_known_args()


if __name__ == '__main__':
    flags, _ = parse_args()
    num_split = flags.num_split
    num_idx = flags.num_idx
    #show_data(data_list)
    if flags.mode == 'image':
      data_list = get_data(flags.dataset_dir, flags.split_name, 
          flags.mode, 1, 0)
      transform_img(data_list, flags.split_name)
    else:
      data_list = get_data(flags.dataset_dir, flags.split_name, 
          flags.mode, num_split, num_idx)
      transform_mask(data_list)

