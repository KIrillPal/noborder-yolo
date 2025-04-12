#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append('../utils')
sys.path.append('..')

from utils.interpolate.markup_utils import Mask, load_markup, vis_markup, eternal_dataset_info, is_border_object
from utils.interpolate.refine_markup_by_yolo import mask_iou, mask_ioa


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
                    #canvas = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
                    # if group_id == 7:
                    #     vis_markup(canvas, [obj, obj2], Path(f'test/{idx}_{k}.png'))
                    if mask_ioa(obj, obj2) > IOU_THRESHOLD:
                        matched_groups.append(groups[idx][k])
                idx -= 1
            
            # if group_id == 7:
            #     print(matched_groups)
            
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
                if mask_ioa(obj, other_obj) > IOU_THRESHOLD:
                    is_mask_ioa_matched[i][j] = True
                    # Match class
                    if obj['cls'] == other_obj['cls']:
                        is_ioa_matched[i][j] = True
                    
    return is_matched, is_mask_matched, is_ioa_matched, is_mask_ioa_matched


# In[11]:


def idx_to_groups(group_idx, matching_info, markup, img_paths):
    is_matched, is_mask_matched, is_ioa_matched, is_mask_ioa_matched = matching_info
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
                "is_mask_ioa_matched": is_mask_ioa_matched[i][j]
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
        stream=True
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


# In[ ]:


gt_groups_per_dataset = []
pred_groups_per_dataset = []

from ultralytics import YOLO
model = YOLO(config['models'][MODEL_VERSION])
import torch

conf = find_best_conf(model, config)

for dataset_path in config['interpolated']['datasets']:
    gt_groups, pred_groups = get_gt_and_predicted_groups(model, config, dataset_path, conf)
    gt_groups_per_dataset.append(gt_groups)
    pred_groups_per_dataset.append(pred_groups)


# In[14]:


for group in pred_groups[10:30]:
    matches = [instance['is_matched'] for instance in group]
    
    print([bool(f) for f in matches], np.array(matches).mean())


# In[15]:


import pickle
with open('gt_groups_test_extended.pkl', 'wb') as f:
    pickle.dump(gt_groups, f)
with open('pred_groups_test_extended.pkl', 'wb') as f:
    pickle.dump(pred_groups, f)
with open('gt_groups_per_dataset_test_extended.pkl', 'wb') as f:
    pickle.dump(gt_groups_per_dataset, f)
with open('pred_groups_per_dataset_test_extended.pkl', 'wb') as f:
    pickle.dump(pred_groups_per_dataset, f)


# In[40]:


import pickle
# with open('gt_groups.pkl', 'rb') as f:
#     gt_groups = pickle.load(f)
# with open('pred_groups.pkl', 'rb') as f:
#     pred_groups = pickle.load(f)
with open('gt_groups_per_dataset_test_extended.pkl', 'rb') as f:
    gt_groups_per_dataset = pickle.load(f)
with open('pred_groups_per_dataset_test_extended.pkl', 'rb') as f:
    pred_groups_per_dataset = pickle.load(f)

gt_groups = []
pred_groups = []
for gt_group in gt_groups_per_dataset:
    gt_groups += gt_group
for pred_group in pred_groups_per_dataset:
    pred_groups += pred_group


# In[17]:


# Update border objects
for g in gt_groups:
    for instance in g:
        instance['is_border'] = is_border_object(Mask(instance['obj']), (config['imgsz'], config['imgsz']))
for g in pred_groups:
    for instance in g:
        instance['is_border'] = is_border_object(Mask(instance['obj']), (config['imgsz'], config['imgsz']))


# In[41]:


interp_test_imgs_path = Path(config['interpolated']['test']).parent / 'test' / 'images'
interpolated_test_names = set(p.name for p in interp_test_imgs_path.iterdir())


# Отфильтурем объекты по тем, что встречались в неинтерполированном тестовом датасете

# In[31]:


gt_groups = [[m for m in g if m['img'].name in interpolated_test_names] for g in gt_groups]
gt_groups = [g for g in gt_groups if len(g) > 0]


# In[32]:


pred_groups = [[m for m in g if m['img'].name in interpolated_test_names] for g in pred_groups]
pred_groups = [g for g in pred_groups if len(g) > 0]


# #### Все объекты

# In[42]:


print("Per-mask precision:", np.array([float(m['is_matched']) for g in pred_groups for m in g]).mean())
print("Per-obj precision:", np.array([max(float(m['is_matched']) for m in g) for g in pred_groups]).mean())


# In[43]:


print("Per-mask recall:", np.array([float(m['is_matched']) for g in gt_groups for m in g]).mean())
print("Per-obj recall:", np.array([max(float(m['is_matched']) for m in g) for g in gt_groups]).mean())


# #### Без краевых объектов

# In[44]:


print("Per-mask precision:", np.array([float(m['is_matched']) for g in pred_groups for m in g if not m['is_border']]).mean())
print("Per-obj precision:", np.array([max(float(m['is_matched']) for m in g if not m['is_border']) for g in pred_groups if len([float(m['is_matched']) for m in g if not m['is_border']]) > 0]).mean())


# In[45]:


print("Per-mask recall:", np.array([float(m['is_matched']) for g in gt_groups for m in g if not m['is_border']]).mean())
print("Per-obj recall:", np.array([max([float(m['is_matched']) for m in g if not m['is_border']] + [0]) for g in gt_groups if len([float(m['is_matched']) for m in g if not m['is_border']]) > 0]).mean())


# #### Доля ошибок классификации
# 

# In[ ]:


print("GT per-mask ratio:", np.array([float(m['is_mask_matched']) for g in gt_groups for m in g if not m['is_matched']]).mean())
print("GT per-obj ratio:", np.array([max(float(m['is_mask_matched']) for m in g) for g in gt_groups if max(float(m['is_matched']) for m in g) == False]).mean())


# In[47]:


print("Pred per-mask ratio:", np.array([float(m['is_mask_matched']) for g in pred_groups for m in g if not m['is_matched']]).mean())
print("Pred per-obj ratio:", np.array([max(float(m['is_mask_matched']) for m in g) for g in pred_groups if max(float(m['is_matched']) for m in g) == False]).mean())


# ### Гипотеза о зависимости обрезанности и качества распознавания

# In[69]:


gt_is_matched = []
gt_is_border = []
pred_is_matched = []
pred_is_border = []

for group in gt_groups:
    for obj in group:
        gt_is_matched.append(obj['is_matched'])
        gt_is_border.append(obj['is_border'])
        
for group in pred_groups:
    for obj in group:
        pred_is_matched.append(obj['is_matched'])
        pred_is_border.append(obj['is_border'])

is_matched = {"Pred": pred_is_matched, "GT": gt_is_matched, 'Total': pred_is_matched + gt_is_matched}
is_border = {"Pred": pred_is_border, "GT": gt_is_border, 'Total': pred_is_border + gt_is_border}


# In[70]:


from scipy.stats import spearmanr

print("| Sample | Spearman correlation | p-value               |")
print("|--------|----------------------|-----------------------|")
for sample in ['GT', 'Pred', 'Total']:
    corr, p_value = spearmanr(is_border[sample], is_matched[sample])
    print(f"| {sample:>6} | {corr:>20.6f} | {p_value:>21} |")


# ### Количество объектов, предсказанных только по краевым объектам

# In[159]:


def get_border_only_matched(groups):
    border_only_matched = []
    border_only_matched_ids = []
    for i, group in enumerate(groups):
        border_matched = False
        non_border_matched = False
        non_border_found = False
        for obj in group:
            if obj['is_matched']:
                if obj['is_border']:
                    border_matched = True
                else:
                    non_border_matched = True
            if not obj['is_border']:
                non_border_found = True
        if border_matched and not non_border_matched and non_border_found:
            border_only_matched.append(group)
            border_only_matched_ids.append(i)
    return border_only_matched, border_only_matched_ids


# In[160]:


gt_border_only, gt_border_ids = get_border_only_matched(gt_groups)
pred_border_only, pred_border_ids = get_border_only_matched(pred_groups)


# In[161]:


print("GT:", len(gt_border_only), "elements", "out of", len(gt_groups), f"({len(gt_border_only) / len(gt_groups) * 100:.2f}%)")
print("Pred:", len(pred_border_only), "elements", "out of", len(pred_groups), f"({len(pred_border_only) / len(pred_groups) * 100:.2f}%)")


# In[189]:


idx = 60
print([p['is_border'] for p in gt_border_only[idx]])
print([p['is_matched'] for p in gt_border_only[idx]])


# In[192]:


iidx = 0
gidx = gt_border_ids[idx]


# In[225]:


get_ipython().run_line_magic('matplotlib', 'inline')
import cv2
from matplotlib import pyplot as plt
from utils.interpolate.markup_utils import vis_markup

for instance in gt_groups[gidx][iidx:iidx+1]:
    print(instance['is_matched'], instance['is_border'], instance['img'])
    img = cv2.imread(instance['img'])
    img = vis_markup(img, [instance['obj']])
    fig, axs = plt.subplots(1, 2, figsize=(15, 7))
    axs[0].set_title('Ground truth')
    axs[0].axis('off')
    axs[0].imshow(img)
    
    pred_img_path = pred_images_dir / f'{instance["img"].stem}.jpg'
    img = cv2.imread(pred_img_path)
    axs[1].set_title('Prediction')
    axs[1].axis('off')
    axs[1].imshow(img)
iidx += 1
plt.show()


# In[218]:


iidx = 0
idx += 1
gidx = gt_border_ids[idx]


# In[ ]:




