import numpy as np


class QuadNode:
    """Quadtree node storing spatial bbox, point indices, level, and children."""
    def __init__(self, bbox, indices_src, indices_tgt, level, parent, root=None):
        bbox_dict = {
            "min_bound": bbox.get_min_bound().tolist(),
            "max_bound": bbox.get_max_bound().tolist()
        }
        self.bbox = bbox_dict
        self.center = (bbox.get_min_bound() + bbox.get_max_bound()) / 2
        self.indices_src = indices_src
        self.indices_tgt = indices_tgt
        self.level = level
        self.fitness = -1
        self.inlier_rmse = -1
        self.planarity = -1
        self.global_transform = np.zeros((4, 4))
        self.local_transform = np.zeros((4, 4))
        self.metrics = {}
        self.size = np.min([len(indices_src), len(indices_tgt)])
        self.parent = parent
        self.children = []
        self.is_leaf = True
        self.is_absurd = False

    def __len__(self):
        counter = len(self.children)
        for child in self.children:
            counter += len(child)
        return counter
    
    def get_root(self):
        if self.parent == None:
            return self
        parent = self.parent
        while parent.parent != None:
            parent = parent.parent
        return parent
