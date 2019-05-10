import argparse
import numpy as np
import os
import PIL
import sys
from os import listdir
from os.path import isfile, join
from PIL import Image
from sklearn.model_selection import train_test_split

MAX_HEIGHT=1024
MAX_WIDTH=1024
PEOPLE_TAG = 1

OUTPUT_DIR = 'output'

def split_dataset(filenames, val_ratio):
  train_files, val_files = train_test_split(filenames, test_size=val_ratio)

  print('train: %d, val: %d' % (len(train_files), len(val_files)))
  seg_dir = OUTPUT_DIR + '/segmentation'
  if not os.path.exists(seg_dir):
      os.makedirs(seg_dir)

  with open(seg_dir + '/train.txt', 'w') as f:
    for item in train_files:
      f.write("%s\n" % item)

  with open(seg_dir + '/val.txt', 'w') as f:
    for item in val_files:
      f.write("%s\n" % item)


def transform(dataset_dir):
    output_img_dir = OUTPUT_DIR + '/images'
    output_mask_dir = OUTPUT_DIR + '/masks'
    if not os.path.exists(output_img_dir):
        os.makedirs(output_img_dir)
    if not os.path.exists(output_mask_dir):
        os.makedirs(output_mask_dir)

    img_dir = 'img'
    mask_dir = 'masks_machine'
    img_dir = os.path.join(dataset_dir, img_dir)
    mask_dir = os.path.join(dataset_dir, mask_dir)
    filenames = [f for f in listdir(img_dir) if isfile(join(img_dir, f))]
    img_count = 0
    for filename in filenames:
        img_path = os.path.join(img_dir, filename)
        name, extension = os.path.splitext(filename)
        out_img_filename = name + '.jpg'
        if extension == ".jpeg":
          filename = name + '.png'
        mask_path = os.path.join(mask_dir, filename)
        if not os.path.exists(mask_path):
          raise Exception('Mask not exist: %s' % mask_path)

        org_img = Image.open(img_path)
        mask_img = Image.open(mask_path)
        if org_img.size != mask_img.size:
            print('Size not match:', img_path)
            continue

        img_size = org_img.size # [width, height]                               
        # scale the image to reasonable size
        if img_size[0] > MAX_WIDTH or img_size[1] > MAX_HEIGHT:                 
            scale = min(MAX_WIDTH * 1.0  / img_size[0], MAX_HEIGHT * 1.0  / img_size[1])
            sw = int(img_size[0] * scale)                                       
            sh = int(img_size[1] * scale)                                       
            org_img = org_img.resize((sw, sh), PIL.Image.LANCZOS)               
            mask_img = mask_img.resize((sw, sh))             
        # transform
        mask_img = mask_img.split()[0]
        mask_data = np.asarray(mask_img, dtype=np.uint8).copy()
        org_shape = mask_data.shape
        mask_data = mask_data.reshape(-1)
        mask_data[mask_data > 0] = PEOPLE_TAG 
        mask_data = mask_data.reshape(org_shape)
        mask_img = Image.fromarray(mask_data.astype(dtype=np.uint8))
        # save
        out_img_path = os.path.join(output_img_dir, out_img_filename)
        out_mask_path = os.path.join(output_mask_dir, filename)
        with open(out_img_path, 'w') as f:
            org_img.convert('RGB').save(out_img_path, 'JPEG')
        with open(out_mask_path, 'w') as f:
            mask_img.save(out_mask_path, 'PNG')
        img_count += 1
        print('%s converted' % img_path)
    filenames = [os.path.splitext(name)[0] for name in filenames]
    return filenames


def run(dataset_dir, val_ratio):
    filenames = transform(dataset_dir)
    split_dataset(filenames, val_ratio)
    print('total: %s' % len(filenames))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default='dataset',
                        help="dataset directory")
    parser.add_argument("--val_ratio", type=float, default=0.1,
                        help="validation set ratio")
    return parser.parse_known_args()

if __name__ == "__main__":
    flags, _ = parse_args()
    run(flags.dataset_dir, flags.val_ratio)

