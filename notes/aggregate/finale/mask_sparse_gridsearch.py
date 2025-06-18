#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Load config
import json
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt

import sys
sys.path.append('../../../utils')
sys.path.append('../../..')


# In[2]:


IMAGE_SHAPE = (1000, 1000)
CONFIG_PATH = '../../../config.json'
SPLIT = 'test'
IOU_THRESHOLD = 0.7
USE_BOX = False


# ### Trainval


# In[4]:


import pickle
import json
TEST_NAME = 'sparse'
with open(f'/alpha/projects/wastie/code/kondrashov/delta/groups/gt_test_{TEST_NAME}.pkl', 'rb') as f:
    gt_extended_groups = pickle.load(f)
with open(f'/alpha/projects/wastie/code/kondrashov/delta/groups/pred_test_{TEST_NAME}.pkl', 'rb') as f:
    pred_extended_groups = pickle.load(f)
with open(f'../../../data/{TEST_NAME}_test/speed.json') as f:
    speed_dict = json.load(f)
    shifts = {}
    speed_sum = 0
    for k, v in sorted(speed_dict.items(), key=lambda x: x[0]):
        speed_sum += v
        shifts[Path(k).name] = speed_sum


# In[5]:


confs = np.array([o['obj']['conf'] for g in pred_extended_groups for o in g])


# ### Получим группы на ленте

# In[6]:


from copy import deepcopy
import random
from utils.integrate.integrate import shift_mask
from utils.interpolate.refine_markup_by_yolo import poly, mask_iou, get_box

def make_box_from_mask(obj):
    box = get_box(obj)
    x1 = [box[0], box[1]]
    x2 = [box[0], box[3]]
    x3 = [box[2], box[3]]
    x4 = [box[2], box[1]]
    obj['points'] = np.array([x1, x2, x3, x4])
    return obj

def align_group(ext_group, shifts):
    ext_group = deepcopy(ext_group)
    for obj in ext_group:
        if USE_BOX:
            obj['obj'] = make_box_from_mask(obj['obj'])
        shift = shifts[obj['img'].name]
        conf = None
        if 'conf' in obj['obj']:
            conf = float(obj['obj']['conf'])
        obj['obj'] = shift_mask(obj['obj'], shift)
        obj['obj']['is_border'] = obj['is_border']
        if conf is not None:
            obj['obj']['conf'] = conf
    return ext_group

def align_groups(ext_group, shifts):
    return [
        align_group(ext_group, shifts)
        for ext_group in ext_group
    ]


# In[7]:


gt_aligned_groups = align_groups(gt_extended_groups, shifts)
pred_aligned_groups = align_groups(pred_extended_groups, shifts)


# In[8]:


gt_groups = [[o['obj'] for o in g] for g in gt_aligned_groups]
pred_groups = [[o['obj'] for o in g]  for g in pred_aligned_groups]


# In[10]:


print(len(gt_groups), len(pred_groups))


# In[11]:


for polygon in map(poly, gt_groups[random.randint(0, len(gt_groups))]):
    x, y = polygon.exterior.xy
    plt.plot(x, y)
plt.show()


# ### Создадим агрегатор

# In[12]:


from src.aggregate.base import Aggregator, ComposeClassifier
from src.aggregate.threshold import ThresholdClassifier
from src.aggregate.noborder import NoBorderClassifier
from src.aggregate.confidence import MaxConfidenceClassifier, ConfidenceClassifier
from src.aggregate.merge import UnionMerger, NMSMerger, SoftNMSMerger, WeightedBoxFusionMerger


# In[13]:


N_CLASSES = 2


# In[14]:


or_classifier = ThresholdClassifier(N_CLASSES, 1)
gt_merger = UnionMerger(N_CLASSES)
gt_aggregator = Aggregator(or_classifier, gt_merger)


# ### Агрегация

# In[15]:


gt = gt_aggregator.aggregate(gt_groups)


# ### Метрики

# In[16]:


from tqdm import tqdm 

def boxes_intersect(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # Check if one box is to the right of the other
    if x1 + w1 <= x2 or x2 + w2 <= x1:
        return False
        
    # Check if one box is above the other
    if y1 + h1 <= y2 or y2 + h2 <= y1:
        return False
        
    return True


# In[17]:


import numpy as np
from shapely.strtree import STRtree
from shapely.geometry import Polygon, box
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import multiprocessing as mp

def get_box(points):
    """Вычисление ограничивающего прямоугольника для списка точек"""
    points = np.array(points)
    return [np.min(points[:, 0]), np.min(points[:, 1]), 
            np.max(points[:, 0]), np.max(points[:, 1])]

def mask_iou_from_points(gt_points, pred_points):
    """Вычисление IoU для двух масок по их точкам"""
    gt_poly = Polygon(gt_points)
    pred_poly = Polygon(pred_points)
    
    if not gt_poly.is_valid or not pred_poly.is_valid:
        return 0.0
    
    intersection = gt_poly.intersection(pred_poly).area
    union = gt_poly.union(pred_poly).area
    
    return intersection / union if union > 0 else 0.0

def process_chunk(args):
    """Обработка пакета данных для параллельных вычислений"""
    chunk, iou_threshold = args
    chunk_results = []
    for i, j, gt_points, pred_points in chunk:
        iou = mask_iou_from_points(gt_points, pred_points)
        if iou > iou_threshold:
            chunk_results.append((i, j))
    return chunk_results

def find_matches(gt: list[dict], pred: list[dict], iou_threshold: float = 0.7):
    """Поиск совпадений между GT и предсказанными масками с использованием IoU"""
    # Извлечение точек и вычисление bounding boxes
    gt = [obj for obj in gt if len(obj['points']) > 0]
    pred = [obj for obj in pred if len(obj['points']) > 0]
    gt_points_list = [np.array(obj['points']) for obj in gt]
    pred_points_list = [np.array(obj['points']) for obj in pred]
    
    gt_boxes = [get_box(points) for points in gt_points_list]
    pred_boxes = [get_box(points) for points in pred_points_list]
    
    # Создание геометрий для пространственного индекса
    gt_geoms = [box(*b) for b in gt_boxes]
    pred_geoms = [box(*b) for b in pred_boxes]
    
    # Построение пространственного индекса для быстрого поиска пересечений
    tree = STRtree(pred_geoms)
    pairs_indices = []
    
    # Поиск пересекающихся пар с использованием пространственного индекса
    for i, gt_geom in enumerate(gt_geoms):
        j_indices = tree.query(gt_geom)
        for j in j_indices:
            pairs_indices.append((i, j))
    
    # Подготовка данных для параллельной обработки
    tasks = []
    chunk_size = min(1000, max(1, len(pairs_indices) // (mp.cpu_count() * 4)))
    
    for idx in range(0, len(pairs_indices), chunk_size):
        chunk = []
        for i, j in pairs_indices[idx:idx+chunk_size]:
            chunk.append((
                i, j,
                gt_points_list[i],
                pred_points_list[j]
            ))
        tasks.append((chunk, iou_threshold))
    
    # Параллельное вычисление IoU
    gt_matches = [False] * len(gt)
    pred_matches = [False] * len(pred)
    
    if tasks:
        with ProcessPoolExecutor() as executor:
            results = list(tqdm(
                executor.map(process_chunk, tasks),
                total=len(tasks),
                desc="Processing pairs"
            ))
            
            for chunk_results in results:
                for i, j in chunk_results:
                    gt_matches[i] = True
                    pred_matches[j] = True
    
    return np.array(gt_matches), np.array(pred_matches)


# ### Перебор

# In[32]:


import numpy as np
from multiprocessing import Pool, cpu_count, Process, Manager
from functools import partial
from tqdm import tqdm
import pickle

def process_combination(args, gt, pred_groups, N_CLASSES, IMAGE_SHAPE, IOU_THRESHOLD):
    remove_border, input_conf, threshold, max_conf, agg_type = args
    
    # Пропускаем невалидные комбинации
    if max_conf < input_conf:
        return None
    
    # Создаем классификаторы
    input_conf_filter = ConfidenceClassifier(N_CLASSES, input_conf)
    noborder_classifier = NoBorderClassifier(N_CLASSES, IMAGE_SHAPE)
    threshold_classifier = ThresholdClassifier(N_CLASSES, threshold)
    confidence_classifier = MaxConfidenceClassifier(N_CLASSES, max_conf)
    
    classifier = ComposeClassifier([
        input_conf_filter,
        *([noborder_classifier] if remove_border else []),
        threshold_classifier,
        confidence_classifier
    ])
    
    if agg_type == 'nms':
        merger = NMSMerger(N_CLASSES, IOU_THRESHOLD)
    elif agg_type == 'soft-nms-3':
        merger = SoftNMSMerger(N_CLASSES, conf_threshold=0.3)
    elif agg_type == 'soft-nms-5':
        merger = SoftNMSMerger(N_CLASSES, conf_threshold=0.5)
    elif agg_type == 'soft-nms-7':
        merger = SoftNMSMerger(N_CLASSES, conf_threshold=0.7)
    elif agg_type == 'union':
        merger = UnionMerger(N_CLASSES)
    elif agg_type == 'wbf':
        merger = WeightedBoxFusionMerger(N_CLASSES)
    # Агрегируем предсказания
    aggregator = Aggregator(classifier, merger)
    pred = aggregator.aggregate(pred_groups)
    
    # Вычисляем метрики
    gt_matches, pred_matches = find_matches(gt, pred, IOU_THRESHOLD)
    
    precision = float(np.mean(pred_matches) if len(pred_matches) > 0 else 0)
    recall = float(np.mean(gt_matches) if len(gt_matches) > 0 else 0)
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
    
    return (("mask", agg_type, remove_border, float(input_conf), threshold, float(max_conf)), (f1, precision, recall))

def parallel_parameter_search(gt, pred_groups, N_CLASSES, IMAGE_SHAPE, IOU_THRESHOLD):
    thresholds = list(range(1, 11))
    input_confidences = np.linspace(0.5, 0.95, 10)
    max_confidences = np.linspace(0.5, 0.95, 10)
    remove_border_options = [False, True]
    
    agg_types = ["nms", "soft-nms-3", "soft-nms-5", "soft-nms-7", "union"]
    if USE_BOX:
        agg_types.append("wbf")

    # Генерируем все комбинации параметров
    param_combinations = [
        (remove_border, input_conf, threshold, max_conf, agg_type)
        for remove_border in remove_border_options
        for input_conf in input_confidences
        for threshold in thresholds
        for max_conf in max_confidences
        for agg_type in agg_types
        if max_conf >= input_conf
    ]
    
    results = Manager().dict()
    
    # Создаем и запускаем процессы
    processes = []
    num_processes = cpu_count()
    chunk_size = len(param_combinations) // num_processes
    
    for i in range(num_processes):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size if i < num_processes - 1 else len(param_combinations)
        process_combinations = param_combinations[start_idx:end_idx]
        
        def process_combinations_worker(combinations):
            for combo in tqdm(combinations, desc=f"Process {i}"):
                result = process_combination(combo,
                    gt=gt, 
                    pred_groups=pred_groups,
                    N_CLASSES=N_CLASSES,
                    IMAGE_SHAPE=IMAGE_SHAPE,
                    IOU_THRESHOLD=IOU_THRESHOLD
                )
                if result is not None:
                    key, metrics = result
                    results.update({key: metrics})
                else:
                    print("result is None")
                    
        process = Process(target=process_combinations_worker, args=(process_combinations,))
        
        processes.append(process)
        process.start()
    
    # Ждем завершения всех процессов
    for process in processes:
        process.join()
    
    return results


if __name__ == '__main__':
    results = parallel_parameter_search(gt, pred_groups, N_CLASSES, IMAGE_SHAPE, IOU_THRESHOLD)
    print(results)
    with open(f'mask_{TEST_NAME}_gridsearch.pkl', 'wb') as f:
        pickle.dump(dict(results), f)

