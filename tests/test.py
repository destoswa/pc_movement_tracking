# dependencies
import os
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import pickle
from tqdm import tqdm
import open3d as o3d
import laspy
from plyfile import PlyData, PlyElement
import rasterio
from rasterio.transform import from_origin
from copy import deepcopy

import geopandas as gpd
from shapely.geometry import Point
import pandas as pd


def get_translate_from_element(el):
    bbox_dict = el[1][0]
    transform = el[1][3]

    xmin, ymin, zmin = bbox_dict['min_bound']
    xmax, ymax, zmax = bbox_dict['max_bound']
    center = np.vstack([np.array([xmax + xmin, ymax + ymin, zmax + zmin]).reshape((3,1)) / 2, np.array([1])])
    translated = np.linalg.matmul(transform, center)
    norm = float(np.linalg.norm(translated - center))

    return norm

if __name__ == "__main__":
    # laoding of gpkg file
    # src_gpkg = r"D:\GitHubProjects\Terranum_repo\pc_movement_tracking\data\test_5_min_points\3000_max_lvl_7\results\points_translate.gpkg"
    src_transforms = r"D:\GitHubProjects\Terranum_repo\pc_movement_tracking\data\test_5_min_points\3000_max_lvl_7\results\pyramid_transforms_test.pickle"
    with open(src_transforms, 'rb') as f:
        lst_transforms = pickle.load(f)
    # data = gpd.read_file(src_gpkg)
    # data['is_absurd'] = np.zeros(len(data), dtype=np.bool)
    lst_absurds = np.zeros(len(lst_transforms), dtype=np.bool)

    # loop over nodes:
    for id_el, el in tqdm(enumerate(lst_transforms), total=len(lst_transforms)):
        # print(id_el)
        lvl_el = el[0]

        if lvl_el == 0:
            continue
        # compare to parent value
        val = get_translate_from_element(el)
        id_parent = id_el-1
        
        while(lst_transforms[id_parent][0] != lvl_el-1):
            id_parent -= 1

        parent_val = get_translate_from_element(lst_transforms[id_parent])
        diff = abs(parent_val - val)
        if diff > 10:
            print(diff)
            lst_absurds[id_el]

    # if absurde, set the other correct siblings as co

    


