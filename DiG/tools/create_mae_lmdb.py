import os
import lmdb # install lmdb by "pip install lmdb"
import cv2
import numpy as np
from tqdm import tqdm
import six
from PIL import Image
import scipy.io as sio
from tqdm import tqdm
import re

def checkImageIsValid(imageBin):
  if imageBin is None:
    return False
  imageBuf = np.fromstring(imageBin, dtype=np.uint8)
  # imageBuf = np.frombuffer(imageBin, dtype=np.uint8)
  if imageBuf.size == 0:
    return False
  img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
  imgH, imgW = img.shape[0], img.shape[1]
  if imgH * imgW == 0:
    return False
  return True


def writeCache(env, cache):
  with env.begin(write=True) as txn:
    for k, v in cache.items():
      txn.put(k.encode(), v)


def _is_difficult(word):
  assert isinstance(word, str)
  return not re.match('^[\w]+$', word)


def createDataset(outputPath, imagePathList, labelList, lexiconList=None, checkValid=True):
  """
  Create LMDB dataset for CRNN training.
  ARGS:
      outputPath    : LMDB output path
      imagePathList : list of image path
      labelList     : list of corresponding groundtruth texts
      lexiconList   : (optional) list of lexicon lists
      checkValid    : if true, check the validity of every image
  """
  if not os.path.exists(os.path.dirname(outputPath)):
    os.makedirs(os.path.dirname(outputPath))

  assert(len(imagePathList) == len(labelList))
  nSamples = len(imagePathList)
  env = lmdb.open(outputPath, map_size=1099511627776)
  cache = {}
  cnt = 1
  for i in tqdm(range(nSamples)):
    imagePath = imagePathList[i]
    label = labelList[i]
    if len(label) == 0:
      continue
    if not os.path.exists(imagePath):
      print('%s does not exist' % imagePath)
      continue
    with open(imagePath, 'rb') as f:
      imageBin = f.read()
    if checkValid:
      if not checkImageIsValid(imageBin):
        print('%s is not a valid image' % imagePath)
        continue

    imageKey = 'image-%09d' % cnt
    labelKey = 'label-%09d' % cnt
    cache[imageKey] = imageBin
    cache[labelKey] = label.encode()
    if lexiconList:
      lexiconKey = 'lexicon-%09d' % cnt
      cache[lexiconKey] = ' '.join(lexiconList[i])
    if cnt % 1000 == 0:
      writeCache(env, cache)
      cache = {}
      print('Written %d / %d' % (cnt, nSamples))
    cnt += 1
  nSamples = cnt-1
  cache['num-samples'] = str(nSamples).encode()
  writeCache(env, cache)
  print('Created dataset with %d samples' % nSamples)

if __name__ == "__main__":
  image_root_dir = '/home/ymk-wh/workspace/researches/mkyang/MAE_OCR_Seqclr/output/vis_pretrain_real_image_smallvit_mim_moco_temp0.2_pathTrans_winds4_warmup0_500_lw1_0.1_onlyOriMim_3epochs/'

  dataset_names = ['IIIT5K_3000', 'cocotextval_9896', 'ctw_1572', 'cute80_288', 'ic03_867', 'IC13_857', 'ic13_1015', 'ic15_1811', 'ost_heavy', 'ost_weak', 'svt_647', 'svt_p_645', 'totaltext_2201']

  mae_image_splits = ['mae_image_1', 'mae_image_2', 'mae_image_3', 'mae_image_4', 'mae_image_5']

  for mae_image_split in mae_image_splits:
    split_image_dir = os.path.join(image_root_dir, mae_image_split)

    for dataset_name in dataset_names:
      image_path_list, label_list = [], []
      masked_image_path_list, masked_label_list = [], []

      image_dir = os.path.join(split_image_dir, dataset_name)

      gt_info_path = os.path.join(image_dir, 'info.npy')
      masked_gt_info_path = os.path.join(image_dir, 'masked_info.npy')
      lmdb_output_path = os.path.join(image_root_dir, 'mae_lmdbs', mae_image_split, dataset_name)
      masked_lmdb_output_path = os.path.join(image_root_dir, 'masked_mae_lmdbs', mae_image_split, dataset_name)

      gt_info_dict = np.load(gt_info_path, allow_pickle=True)
      gt_info_dict = gt_info_dict.item()
      masked_gt_info_dict = np.load(masked_gt_info_path, allow_pickle=True)
      masked_gt_info_dict = masked_gt_info_dict.item()

      for rel_image_path, label in gt_info_dict.items():
        image_path = os.path.join(image_dir, os.path.basename(rel_image_path))
        image_path_list.append(image_path)
        label_list.append(label)
      createDataset(lmdb_output_path, image_path_list, label_list)

      for rel_image_path, label in masked_gt_info_dict.items():
        image_path = os.path.join(image_dir, os.path.basename(rel_image_path))
        masked_image_path_list.append(image_path)
        masked_label_list.append(label)
      createDataset(masked_lmdb_output_path, masked_image_path_list, masked_label_list)