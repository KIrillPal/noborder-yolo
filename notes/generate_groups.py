#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append('../utils')
sys.path.append('..')

from utils.interpolate.markup_utils import Mask, load_markup, vis_markup, eternal_dataset_info, is_border_object
from utils.interpolate.refine_markup_by_yolo import mask_iou, mask_ioa, poly_mask_area


# ### Загрузка датасета и модели

# In[2]:


CONFIG_PATH = '../config.json'
SPLIT = 'test'
IOU_THRESHOLD = 0.7


# In[3]:


# Load config
import json
import numpy as np
from pathlib import Path


# ### Подготавливаем данные

# In[4]:


# Create temporary directory for predictions
pred_labels_dir = Path('runs/segment/predict/labels')
pred_images_dir = Path('runs/segment/predict')


# In[5]:


from utils.integrate.integrate import shift_mask


# In[6]:


with open(CONFIG_PATH, 'r') as f:
    config = json.load(f)


# In[7]:


def restrict(obj):
    obj = obj.copy()
    obj['points'] = np.clip(obj['points'], 0, config['imgsz'])
    return obj


# In[8]:


from tqdm import tqdm
from statistics import mode

def get_groups(markups, speed, imgsz):
    groups = [
        [-1] * len(m) for m in markups
    ]
    group_id = 0
    for i, markup in enumerate(tqdm(markups)):
        for j, obj in enumerate(markup):
            if groups[i][j] != -1:
                continue
            obj = restrict(obj)
            matched_groups = []
            # Find if group already exists
            idx = i-1
            shift = 0
            while idx >= 0 and shift < 0.5 * imgsz:
                shift += speed[idx + 1]
                for k, other_obj in enumerate(markups[idx]):
                    obj2 = restrict(shift_mask(other_obj, -shift))
                    if mask_ioa(obj, obj2) > IOU_THRESHOLD:
                        matched_groups.append(groups[idx][k])
                idx -= 1
            
            if len(matched_groups) > 2:
                groups[i][j] = mode(matched_groups)
            else:
                groups[i][j] = group_id
                group_id += 1

            # Find next elements from this group
            idx = i+1
            shift = 0
            while idx < len(markups) and shift < 1.2 * imgsz:
                shift += speed[idx]
                for k, other_obj in enumerate(markups[idx]):
                    if groups[idx][k] != -1:
                        continue
                    if mask_ioa(obj, restrict(shift_mask(other_obj, shift))) > IOU_THRESHOLD:
                        groups[idx][k] = groups[i][j]
                idx += 1
    return groups


# In[9]:


OLD_CLS_TO_NEW_INDICES = {'0': '0', '1': '0', '2': '1'}
def cls_transform(markup):
    result = []
    for m in markup.copy():
        m['cls'] = OLD_CLS_TO_NEW_INDICES[m['cls']]
        result.append(m)
    return result


# In[ ]:


from tqdm import tqdm
def get_matches(markups, other):
    is_matched = [
        [False] * len(m) for m in markups
    ]
    is_mask_matched = [
        [False] * len(m) for m in markups
    ]
    is_ioa_matched = [
        [False] * len(m) for m in markups
    ]
    is_mask_ioa_matched = [
        [False] * len(m) for m in markups
    ]
    possible_cls = [
        [set()] * len(m) for m in markups
    ]
    for i, markup in enumerate(tqdm(markups)):
        for j, obj in enumerate(markup):
            obj = restrict(obj)
            for other_obj in other[i]:
                other_obj = restrict(other_obj)
                # Match mask
                if mask_iou(obj, other_obj) > IOU_THRESHOLD:
                    is_mask_matched[i][j] = True
                    # Match class
                    if obj['cls'] == other_obj['cls']:
                        is_matched[i][j] = True
                    else:
                        possible_cls[i][j].add(other_obj['cls'])
                # if mask_ioa(obj, other_obj) > IOU_THRESHOLD:
                #     is_mask_ioa_matched[i][j] = True
                #     # Match class
                #     if obj['cls'] == other_obj['cls']:
                #         is_ioa_matched[i][j] = True
                    
    return is_matched, is_mask_matched, is_ioa_matched, is_mask_ioa_matched, possible_cls


# In[ ]:


def idx_to_groups(group_idx, matching_info, markup, img_paths):
    is_matched, is_mask_matched, is_ioa_matched, is_mask_ioa_matched, possible_cls = matching_info
    groups = [[] for i in range(max(y for x in group_idx for y in x) + 1)]
    for i, group in enumerate(group_idx):
        for j, g in enumerate(group):
            img_path = img_paths[i]
            groups[g].append({
                "img": img_path, 
                "obj": markup[i][j], 
                "is_border": is_border_object(Mask(markup[i][j]), (config['imgsz'], config['imgsz'])), 
                "is_matched": is_matched[i][j],
                "is_mask_matched": is_mask_matched[i][j],
                "is_ioa_matched": is_ioa_matched[i][j],
                "is_mask_ioa_matched": is_mask_ioa_matched[i][j],
                "possible_cls": possible_cls[i][j],
            })
    return groups


# ### Предсказание с лучшим по F1 confidence

# In[12]:


MODEL_VERSION = 'default'


# In[13]:


import shutil

def find_best_conf(model, config):
    # Run validation to get best confidence threshold
    val_results = model.val(data=config['data'], split=SPLIT, device='cuda:1')

    best_f1_idx = np.argmax(val_results.seg.curves_results[1][1].mean(axis=0))
    best_f1 = val_results.seg.curves_results[1][1][..., best_f1_idx].mean()
    best_conf = val_results.seg.curves_results[1][0][best_f1_idx]
    print(f"Best F1: {best_f1:.4f} at confidence {best_conf:.4f}")
    return best_conf


def get_gt_and_predicted_groups(model, config, dataset_path, conf):

    # Load labels
    dataset_path = Path(dataset_path)
    dataset_info = eternal_dataset_info(dataset_path)
    gt_labels_dir = Path(dataset_path) / 'gt_interp'
    
    shutil.rmtree('runs/segment', ignore_errors=True)
    # Run YOLO validation to get the best confidence score

    # Run prediction with best confidence
    for pred in model.predict(
        source=str(dataset_path / 'imgs'),
        conf=conf,
        save_txt=True,
        device='cuda:1',
        save=True,
        stream=True,
        save_conf=True,
    ):
        pass
    
    gt = []
    pred = []
    speed = []
    gt_paths = sorted(list(gt_labels_dir.glob("*.txt")))
    for gt_path in gt_paths:
        pred_path = pred_labels_dir / gt_path.name
        if not pred_path.exists():
            pred_path.touch()
        gt.append(cls_transform(load_markup(gt_path, config['imgsz'])))
        pred.append(load_markup(pred_path, config['imgsz']))
        speed.append(dataset_info['speed'][str(dataset_path / 'imgs' / (gt_path.stem + '.jpg'))])

    gt_img_paths = [p.parent.parent / 'imgs' / f'{p.stem}.jpg' for p in gt_paths]
    pred_img_paths = [pred_images_dir / f'{p.stem}.jpg' for p in gt_paths]

    gt_group_idx = get_groups(gt, speed, config['imgsz'])
    gt_matching_info = get_matches(gt, pred)
    gt_groups = idx_to_groups(gt_group_idx, gt_matching_info, gt, gt_img_paths)
    
    pred_group_idx = get_groups(pred, speed, config['imgsz'])
    pred_matching_info = get_matches(pred, gt)
    pred_groups = idx_to_groups(pred_group_idx, pred_matching_info, pred, pred_img_paths)
    
    return gt_groups, pred_groups


# In[14]:


# gt_groups_per_dataset = []
# pred_groups_per_dataset = []

# from ultralytics import YOLO
# model = YOLO(config['models'][MODEL_VERSION])
# import torch

# conf = find_best_conf(model, config)

# for dataset_path in config['interpolated']['datasets']:
#     gt_groups, pred_groups = get_gt_and_predicted_groups(model, config, dataset_path, conf)
#     gt_groups_per_dataset.append(gt_groups)
#     pred_groups_per_dataset.append(pred_groups)


# In[19]:


CHOSEN_DATASETS = [
    # "/alpha/projects/wastie/eternal-storage/dataset/tula_sep_0001_2024_06_21_11_00",
    "/alpha/projects/wastie/eternal-storage/dataset/tula_sep_0001_2024_06_21_12_00",
    # "/alpha/projects/wastie/eternal-storage/dataset/tula_sep_0001_2024_06_21_13_00",
    # "/alpha/projects/wastie/eternal-storage/dataset/tula_sep_0001_2024_07_16_14_13",
    # "/alpha/projects/wastie/eternal-storage/dataset/tula_sep_0001_2024_07_16_14_43",
    # "/alpha/projects/wastie/eternal-storage/dataset/tula_sep_0001_2024_07_16_15_14",
    # "/alpha/projects/wastie/eternal-storage/dataset/tula_sep_0001_2024_07_16_15_44",
    # "/alpha/projects/wastie/eternal-storage/dataset/tula_sep_0001_2024_07_16_16_14",
    # "/alpha/projects/wastie/eternal-storage/dataset/tula_sep_0001_2024_08_08_11_36",
    # "/alpha/projects/wastie/eternal-storage/dataset/tula_sep_0001_2024_08_08_12_06",
    # "/alpha/projects/wastie/eternal-storage/dataset/tula_sep_0001_2024_08_08_12_37",
    # "/alpha/projects/wastie/eternal-storage/dataset/tula_sep_0001_2024_08_12_10_51",
    # "/alpha/projects/wastie/eternal-storage/dataset/tula_sep_0001_2024_08_12_11_21",
    # "/alpha/projects/wastie/eternal-storage/dataset/tula_sep_0001_2024_08_12_12_21",
    # "/alpha/projects/wastie/eternal-storage/dataset/tula_sep_0001_2024_08_20_10_10",
    # "/alpha/projects/wastie/eternal-storage/dataset/tula_sep_0001_2024_08_20_10_40"
]


# 

# In[ ]:


gt_groups_per_dataset = []
pred_groups_per_dataset = []

from ultralytics import YOLO
model = YOLO(config['models'][MODEL_VERSION])
import torch

conf = find_best_conf(model, config)

for dataset_path in CHOSEN_DATASETS:
    gt_groups, pred_groups = get_gt_and_predicted_groups(model, config, dataset_path, conf)
    gt_groups_per_dataset.append(gt_groups)
    pred_groups_per_dataset.append(pred_groups)


# 

# In[21]:


# Update border objects and calculate area
for g in gt_groups:
    for instance in g:
        instance['is_border'] = is_border_object(Mask(instance['obj']), (config['imgsz'], config['imgsz']))
        instance['area'] = poly_mask_area(instance['obj'])
for g in pred_groups:
    for instance in g:
        instance['is_border'] = is_border_object(Mask(instance['obj']), (config['imgsz'], config['imgsz']))
        instance['area'] = poly_mask_area(instance['obj'])


# In[22]:


# Calculate area share based on the group's max area
for groupset in [gt_groups, pred_groups]:
    for g in groupset:
        max_area = max([instance['area'] for instance in g])
        for instance in g:
            instance['area_share'] = instance['area'] / max_area


# In[ ]:


# import pickle
# with open('gt_noborder_aug_trainval.pkl', 'wb') as f:
#     pickle.dump(gt_groups, f)
# with open('pred_noborder_aug_trainval.pkl', 'wb') as f:
#     pickle.dump(pred_groups, f)


# In[28]:


import pickle
with open('gt_dense_trainval.pkl', 'wb') as f:
    pickle.dump(gt_groups, f)
with open('gt_dense_trainval.pkl', 'wb') as f:
    pickle.dump(pred_groups, f)
with open('gt_dense_trainval_per_dataset.pkl', 'wb') as f:
    pickle.dump(gt_groups_per_dataset, f)
with open('pred_dense_trainval_per_dataset.pkl', 'wb') as f:
    pickle.dump(pred_groups_per_dataset, f)