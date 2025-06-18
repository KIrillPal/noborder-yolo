from shapely.ops import unary_union
from shapely.geometry import MultiPolygon, Polygon, box
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict

from utils.interpolate.refine_markup_by_yolo import poly, pts_from_poly
from src.aggregate.base import BaseMerger, NO_OBJECT

class FirstMerger(BaseMerger):
    def __init__(self, n_classes: int):
        super().__init__(n_classes)

    def merge(self, groups: List[List[Dict]], classes: List[int]):
        return [
            {
                "points": self.merge_group_(g, i),
                "cls": c,
                "count": 1,
                "group_id": i
            } for i, (g, c) in enumerate(zip(groups, classes))
            if c != NO_OBJECT
        ]

    def merge_group_(self, group, group_id):
        for obj in group:
            polygon = poly(obj)
            if isinstance(polygon, Polygon):
                return pts_from_poly(polygon)
        return []

class UnionMerger(BaseMerger):
    def __init__(self, n_classes: int):
        super().__init__(n_classes)
        # self.CACHE = {}

    def merge(self, groups: List[List[Dict]], classes: List[int]):
        return [
            {
                "points": self.merge_group_(g, i),
                "cls": c,
                "count": 1,
                "group_id": i
            } for i, (g, c) in enumerate(zip(groups, classes))
            if c != NO_OBJECT
        ]

    def merge_group_(self, group, group_id):
        # if str(group) in self.CACHE:
        #     return self.CACHE[str(group)]
        
        mask_polygons = [poly(g) for g in group]
        polygon = unary_union(mask_polygons)
        if not isinstance(polygon, Polygon):
            # Get max polygon if there are several
            aviable_polygons = [p for p in polygon.geoms if p.geom_type == 'Polygon']
            polygon = max(aviable_polygons, key=lambda x: x.area)
        result = pts_from_poly(polygon)
        
        # self.CACHE[str(group)] = result 
        return result


class NMSMerger(BaseMerger):
    def __init__(self, n_classes: int, iou_threshold: float = 0.5):
        super().__init__(n_classes)
        self.iou_threshold = iou_threshold

    def merge(self, groups: List[List[Dict]], classes: List[int]):
        results = []
        for group_id, (group, cls) in enumerate(zip(groups, classes)):
            if cls == NO_OBJECT:
                continue
                
            polygons = []
            confs = []
            for item in group:
                polygons.append(poly(item))
                confs.append(item.get("conf", 1.0))
            
            if not polygons:
                continue
                
            # Apply NMS for masks
            keep_indices = self.mask_nms(polygons, confs)
            
            # Keep only the selected masks
            for idx in keep_indices:
                results.append({
                    "points": pts_from_poly(polygons[idx]),
                    "cls": cls,
                    "count": len(keep_indices),
                    "group_id": group_id,
                    "conf": confs[idx]
                })
        
        return results

    def mask_nms(self, polygons: List[Polygon], confs: List[float]) -> List[int]:
        """Non-Maximum Suppression for masks using shapely Polygons"""
        if not polygons:
            return []
            
        # Sort by conf
        indices = np.argsort(confs)[::-1]
        keep = []
        
        while indices.size > 0:
            current = indices[0]
            keep.append(current)
            
            # Compute IoU with remaining polygons
            ious = []
            for i in indices[1:]:
                intersection = polygons[current].intersection(polygons[i]).area
                union = polygons[current].union(polygons[i]).area
                iou = intersection / union if union > 0 else 0
                ious.append(iou)
            
            # Filter based on IoU threshold
            indices = indices[1:][np.array(ious) < self.iou_threshold]
        
        return keep


class SoftNMSMerger(BaseMerger):
    def __init__(self, n_classes: int, sigma: float = 0.5, conf_threshold: float = 0.01):
        super().__init__(n_classes)
        self.sigma = sigma
        self.conf_threshold = conf_threshold

    def merge(self, groups: List[List[Dict]], classes: List[int]):
        results = []
        for group_id, (group, cls) in enumerate(zip(groups, classes)):
            if cls == NO_OBJECT:
                continue
                
            # Convert masks to shapely Polygons with confs
            polygons = []
            confs = []
            for item in group:
                poly_obj = poly(item)
                if poly_obj.is_valid and isinstance(poly_obj, Polygon):
                    polygons.append(poly_obj)
                    confs.append(item.get("conf", 1.0))
            
            if not polygons:
                continue
                
            # Apply Soft-NMS for masks
            keep_indices, updated_confs = self.mask_soft_nms(polygons, confs)
            
            # Keep only the selected masks with updated confs
            for idx in keep_indices:
                if updated_confs[idx] >= self.conf_threshold:
                    results.append({
                        "points": pts_from_poly(polygons[idx]),
                        "cls": cls,
                        "count": len(keep_indices),
                        "group_id": group_id,
                        "conf": updated_confs[idx]
                    })
        
        return results

    def mask_soft_nms(self, polygons: List[Polygon], confs: List[float]) -> Tuple[List[int], List[float]]:
        """Corrected Soft Non-Maximum Suppression for masks using shapely Polygons"""
        if not polygons:
            return [], []
            
        confs = np.array(confs, dtype=np.float32)
        indices = np.arange(len(polygons))
        keep = []
        
        while len(indices) > 0:
            # Find polygon with highest conf in current indices
            current = np.argmax(confs[indices])
            best_idx = indices[current]
            keep.append(best_idx)
            
            # Update confs for other polygons in current indices
            for i in indices:
                if i == best_idx:
                    continue
                    
                # Calculate IoU between best_idx and current polygon i
                inter = polygons[best_idx].intersection(polygons[i]).area
                union = polygons[best_idx].union(polygons[i]).area
                iou = inter / union if union > 0 else 0.0
                
                # Apply Gaussian penalty
                confs[i] *= np.exp(-(iou ** 2) / self.sigma)
            
            # Remove best_idx from consideration
            indices = np.delete(indices, current)
            # Filter indices by conf threshold
            mask = confs[indices] >= self.conf_threshold
            indices = indices[mask]
        
        return keep, confs.tolist()


class WeightedBoxFusionMerger(BaseMerger):
    def __init__(self, n_classes: int):
        super().__init__(n_classes)

    def merge(self, groups: List[List[Dict]], classes: List[int]):
        results = []
        for group_id, (group, cls) in enumerate(zip(groups, classes)):
            if cls == NO_OBJECT:
                continue
                
            # Convert masks to boxes with confs
            boxes = []
            confs = []
            polygons = []
            for item in group:
                points = item["points"]
                x1 = min(p[0] for p in points)
                y1 = min(p[1] for p in points)
                x2 = max(p[0] for p in points)
                y2 = max(p[1] for p in points)
                conf = item.get("conf", 1.0)
                boxes.append([x1, y1, x2, y2])
                confs.append(conf)
                polygons.append(poly(item))
            
            if not boxes:
                continue
                
            # Consider all boxes as a single cluster
            weights = np.array(confs)
            boxes_array = np.array(boxes)
            
            # Calculate weighted average of box coordinates
            weighted_boxes = boxes_array * weights[:, np.newaxis]
            fused_box = np.sum(weighted_boxes, axis=0) / np.sum(weights)
            
            # Calculate fused conf
            fused_conf = np.mean(confs)
            
            # Find the best mask in the group (highest conf)
            best_idx = np.argmax(confs)
            best_mask = polygons[best_idx]
            
            results.append({
                "points": pts_from_poly(best_mask),
                "cls": cls,
                "count": len(boxes),
                "group_id": group_id,
                "conf": fused_conf
            })
        
        return results