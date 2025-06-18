#!/usr/bin/env python
# coding: utf-8

# In[1]:


CONFIG_PATH = '../../config.json'
SPLIT = 'test'
IOU_THRESHOLD = 0.7


# In[2]:


# Load config
import json
import numpy as np
from pathlib import Path


# ### Подготавливаем визуализацию

# In[3]:


# Create temporary directory for predictions
pred_labels_dir = Path('runs/segment/predict/labels')
pred_images_dir = Path('runs/segment/predict')


# In[4]:


with open(CONFIG_PATH, 'r') as f:
    config = json.load(f)


# In[5]:


from matplotlib import pyplot as plt


# ## Train-val данные

# In[6]:


import pickle
with open('../../groups/gt_groups_per_dataset_trainval.pkl', 'rb') as f:
    gt_groups_per_dataset = pickle.load(f)
with open('../../groups/pred_groups_per_dataset_trainval.pkl', 'rb') as f:
    pred_groups_per_dataset = pickle.load(f)


# In[7]:


gt_groups = []
pred_groups = []
for gt_group in gt_groups_per_dataset:
    gt_groups += gt_group
for pred_group in pred_groups_per_dataset:
    pred_groups += pred_group


# In[8]:


import sys
sys.path.append('../../utils')
sys.path.append('../..')


# In[9]:


from interpolate.refine_markup_by_yolo import get_box
import cv2
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm


# In[10]:


def make_square_box(box, padding=10):
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1
    
    # Make square by taking larger dimension
    size = max(width, height)
    
    # Calculate center
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    
    # Calculate new box coordinates with padding
    new_x1 = int(center_x - size/2 - padding)
    new_y1 = int(center_y - size/2 - padding)
    new_x2 = int(center_x + size/2 + padding)
    new_y2 = int(center_y + size/2 + padding)
    
    return np.array([new_x1, new_y1, new_x2, new_y2])


# In[11]:


INCLUDED_DATASETS = config['interpolated']['datasets']['train']
PADDING = 0.2


# In[12]:


CACHE = {}
for dataset in tqdm(INCLUDED_DATASETS):
    img_dir = Path(dataset) / 'imgs'
    for i, p in enumerate(sorted(list(img_dir.iterdir()))):
        CACHE[p.stem] = (p, i)

def find_image_by_name(image_name : str):
    return CACHE[image_name]


# In[ ]:


NO_OBJECT = -1
def get_class(obj : dict):
    obj_cls = NO_OBJECT
    if obj['is_matched']:
        obj_cls = int(obj['obj']['cls'])
    elif len(obj['possible_cls']) > 0:
        obj_cls = next(iter(obj['possible_cls']))
    return obj_cls


# In[14]:


import yaml

def extract_groups(groups, output_dir):
    output_dir.mkdir(exist_ok=True)

    for group_idx, group in tqdm(enumerate(groups), total=len(groups)):
        group_dir = output_dir / str(group_idx)
        group_dir.mkdir(exist_ok=True)
        
        meta_path = group_dir / 'meta.yaml'
        group_path = group_dir / 'group.pkl'
        meta = {}

        FIRST_IDX = -1
        for obj_idx, obj in enumerate(group):
            box = get_box(obj['obj'])
            box = np.round(box).astype(int)
            p = round(max(box[2] - box[0], box[3] - box[1]) * PADDING)
            
            box = make_square_box(box, p)
            
            img_path = obj['img']
            obj_cls = get_class(obj)
            
            real_img_path, img_idx = find_image_by_name(img_path.stem)
            img = cv2.imread(str(real_img_path))
            
            padded = cv2.copyMakeBorder(
                img, 
                p, p, p, p, 
                cv2.BORDER_CONSTANT, 
                value=[0,0,0]
            )
            cropped = padded[
                max(p+box[1], 0):
                max(p+box[3], 0), 
                max(p+box[0], 0):
                max(p+box[2], 0)
            ]
            if min(cropped.shape[:2]) < 5:
                print(cropped.shape, p+box[1], p+box[3], p+box[0], p+box[2])
            
            if FIRST_IDX == -1:
                FIRST_IDX = img_idx

            img_filename = f"{img_idx - FIRST_IDX}_{obj_cls}.png"
            output_path = group_dir / img_filename
            cv2.imwrite(str(output_path), cropped)
            
            meta[img_filename] = str(real_img_path)

        with open(meta_path, 'w') as f:
            yaml.dump(meta, f)
        with open(group_path, 'wb') as f:
            pickle.dump(group, f)


# ## Разделение на трейн и валидацию

# In[15]:


from sklearn.model_selection import train_test_split
train_groups, val_groups = train_test_split(pred_groups, test_size=0.15, random_state=42)
print("Train:", len(train_groups))
print("Val:", len(val_groups))


# ## Создание датасета

# In[16]:


output_dir = Path('../../data/train')
extract_groups(train_groups, output_dir)


# In[ ]:


output_dir = Path('../../data/val')
extract_groups(val_groups, output_dir)


# In[ ]:




