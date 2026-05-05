import os
import sys
import numpy as np
import pickle
if __name__ == "__main__":
    sys.path.append(os.getcwd())
from src.quadnode import QuadNode

def compute_translation(root:QuadNode):
    print(root.transform)

def compute_rotations():
    pass


if __name__ == "__main__":
    src_transforms = r"D:\GitHubProjects\Terranum_repo\pc_movement_tracking\data\test_11_min_buffer_area\min_8m\3000_max_lvl_7\results\pyramid_transforms_test.pickle"
    
    with open(src_transforms, 'rb') as f:
        root = pickle.load(f)

    compute_translation(root)