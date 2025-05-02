from shapely.ops import unary_union
from shapely.geometry import MultiPolygon, Polygon, LineString

from utils.interpolate.refine_markup_by_yolo import poly, pts_from_poly
from src.aggregate.base import BaseMerger, NO_OBJECT


class UnionMerger(BaseMerger):
    def __init__(self, n_classes : int):
        super().__init__(n_classes)
        self.max_draw = 10

    def merge(self, groups : list[list[dict]], classes: list[int]):
        return [{
            "points": self.merge_group_(g, i),
            "cls": c,
            "count": 1,
            "group_id": i
        } for i, (g, c) in enumerate(zip(groups, classes))
        if c != NO_OBJECT]


    def merge_group_(self, group, group_id):
        mask_polygons = [poly(g) for g in group]
        polygon = unary_union(mask_polygons)
        if not isinstance(polygon, Polygon):
            # Get max polygon if there are several
            aviable_polygons = [p for p in polygon.geoms if p.geom_type == 'Polygon']
            polygon = max(aviable_polygons, key=lambda x: x.area)
        return pts_from_poly(polygon)