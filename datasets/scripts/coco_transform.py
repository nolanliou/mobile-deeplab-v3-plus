import numpy as np
import os
import argparse
#import matplotlib.pyplot as plt
from PIL import Image
from shutil import copyfile
from pycocotools.coco import COCO

def _real_id_to_cat_id(catId):
  """Note coco has 80 classes, but the catId ranges from 1 to 90!"""
  real_id_to_cat_id = \
    {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 12: 13, 13: 14, 14: 15, 15: 16, 16: 17,
     17: 18, 18: 19, 19: 20, 20: 21, 21: 22, 22: 23, 23: 24, 24: 25, 25: 27, 26: 28, 27: 31, 28: 32, 29: 33, 30: 34,
     31: 35, 32: 36, 33: 37, 34: 38, 35: 39, 36: 40, 37: 41, 38: 42, 39: 43, 40: 44, 41: 46, 42: 47, 43: 48, 44: 49,
     45: 50, 46: 51, 47: 52, 48: 53, 49: 54, 50: 55, 51: 56, 52: 57, 53: 58, 54: 59, 55: 60, 56: 61, 57: 62, 58: 63,
     59: 64, 60: 65, 61: 67, 62: 70, 63: 72, 64: 73, 65: 74, 66: 75, 67: 76, 68: 77, 69: 78, 70: 79, 71: 80, 72: 81,
     73: 82, 74: 84, 75: 85, 76: 86, 77: 87, 78: 88, 79: 89, 80: 90}
  return real_id_to_cat_id[catId]

def _cat_id_to_real_id(readId):
  """Note coco has 80 classes, but the catId ranges from 1 to 90!"""
  cat_id_to_real_id = \
    {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 13: 12, 14: 13, 15: 14, 16: 15, 17: 16,
     18: 17, 19: 18, 20: 19, 21: 20, 22: 21, 23: 22, 24: 23, 25: 24, 27: 25, 28: 26, 31: 27, 32: 28, 33: 29, 34: 30,
     35: 31, 36: 32, 37: 33, 38: 34, 39: 35, 40: 36, 41: 37, 42: 38, 43: 39, 44: 40, 46: 41, 47: 42, 48: 43, 49: 44,
     50: 45, 51: 46, 52: 47, 53: 48, 54: 49, 55: 50, 56: 51, 57: 52, 58: 53, 59: 54, 60: 55, 61: 56, 62: 57, 63: 58,
     64: 59, 65: 60, 67: 61, 70: 62, 72: 63, 73: 64, 74: 65, 75: 66, 76: 67, 77: 68, 78: 69, 79: 70, 80: 71, 81: 72,
     82: 73, 84: 74, 85: 75, 86: 76, 87: 77, 88: 78, 89: 79, 90: 80}
  return cat_id_to_real_id[readId]

def _get_coco_masks(coco, img_id, height, width, img_name):
  """ get the masks for all the instances
  Note: some images are not annotated
  Return:
    masks, mxhxw numpy array
    classes, mx1
    bboxes, mx4
  """
  annIds = coco.getAnnIds(imgIds=[img_id], iscrowd=None)
  if annIds is None or annIds <= 0:
      print('No annotaion ids for %s' % str(img_id))
      return None
  anns = coco.loadAnns(annIds)
  if anns is None or len(anns) == 0:
      print('No annotaion for %s' % str(img_id))
      return None
  #coco.showAnns(anns)
  mask = np.zeros((height, width), dtype=np.float32)
  person_count = 0 
  for ann in anns:
    m = coco.annToMask(ann) # zero one mask
    assert m.shape[0] == height and m.shape[1] == width, \
            'image %s and ann %s dont match' % (img_id, ann)
    cat_id = _cat_id_to_real_id(ann['category_id'])
    if cat_id == 1: # person
        m = m.astype(np.float32) * cat_id
        mask[m > 0] = 1
        person_count += 1 

  mask = mask.astype(np.uint8)
  if person_count > 0 and person_count <= 2: # only use one person
      return mask
  else:
      return None

def transform(mask_dir, ds_dir, anno_dir, split_name, output_dir, vis):
    assert split_name in ['train2017', 'val2017']
    file_prefix = 'coco_'
    out_img_dir = os.path.join(output_dir, 'images')
    out_mask_dir = os.path.join(output_dir, 'masks')
    out_list_dir = os.path.join(output_dir, 'segmentation')
    if not os.path.exists(out_img_dir):
        os.makedirs(out_img_dir)
    if not os.path.exists(out_mask_dir):
        os.makedirs(out_mask_dir)
    if not os.path.exists(out_list_dir):
        os.makedirs(out_list_dir)

    ann_file = os.path.join(anno_dir, 'instances_%s.json' % (split_name))
  
    coco = COCO(ann_file)

    cats = coco.loadCats(coco.getCatIds())
    print('%s has %d images' %(split_name, len(coco.imgs)))
    imgs = [(img_id, coco.imgs[img_id]) for img_id in coco.imgs]
  
    img_size = len(imgs)
    count = 0
    for i in range(img_size):
        img_id = imgs[i][0]
        img_name = imgs[i][1]['file_name']
        img_path = os.path.join(ds_dir, split_name, img_name)
        height, width = imgs[i][1]['height'], imgs[i][1]['width']
        mask = _get_coco_masks(coco, img_id, height, width, img_path)
        if mask is not None:
            new_file_name = file_prefix + os.path.splitext(img_name)[0]
            mask = mask.astype(dtype=np.uint8)
            mask_img = Image.fromarray(mask)
            out_mask_path = os.path.join(out_mask_dir, new_file_name + '.png')
            with open(out_mask_path, 'w') as f:
                mask_img.save(out_mask_path, 'PNG')
            out_img_path = os.path.join(out_img_dir, new_file_name + '.jpg')
            copyfile(img_path, out_img_path)
            # write list file
            if split_name == 'train2017':
              list_file_path = os.path.join(out_list_dir, 'train.txt')
            else:
              list_file_path = os.path.join(out_list_dir, 'val.txt')
            with open(list_file_path, "a") as f:
                f.write(new_file_name + '\n')
            count += 1
            #if vis:
            #  plt.figure(0)
            #  plt.axis('off')
            #  plt.imshow(mask)
            #  plt.show()
    print('total images contains 1 or 2 persons: %d' % count)

def run(dataset_dir, split_name='train2017', vis=False):
    mask_dir = os.path.join(dataset_dir, 'masks')
    annotation_dir = os.path.join(dataset_dir, 'annotations')
    transform(mask_dir, dataset_dir, annotation_dir, split_name, 'output', vis)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vis", help="show images",
                        action="store_true")
    parser.add_argument("--dataset_dir", type=str, default='2017',
                        help="coco dataset directory")
    parser.add_argument("--split", type=str, default='val2017',
                        help="split name[train2017|val2017]")
    return parser.parse_known_args()

if __name__ == "__main__":
    #test()
    flags, _ = parse_args()
    run(flags.dataset_dir, flags.split, flags.vis)
