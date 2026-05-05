import numpy as np


class QuadNode:
    """Quadtree node storing spatial bbox, point indices, level, and children."""
    def __init__(self, bbox, indices_src, indices_tgt,indices_with_neigh, indices_sub_pts, level, parent):
        bbox_dict = {
            "min_bound": bbox.get_min_bound().tolist(),
            "max_bound": bbox.get_max_bound().tolist()
        }
        self.bbox = bbox_dict
        self.indices_src = indices_src
        self.indices_with_neigh = indices_with_neigh
        self.indices_sub_pts = indices_sub_pts
        self.indices_tgt = indices_tgt
        self.level = level
        self.fitness = -1
        self.inlier_rmse = -1
        self.planarity = -1
        self.transform = np.zeros((4,4))
        self.metrics = {}
        self.size = np.min([len(indices_src), len(indices_tgt)])
        self.parent = parent
        self.children = []
        self.is_leaf = True
        self.is_absurd = False