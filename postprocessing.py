# dependencies
import os
import numpy as np
import pickle
import geopandas as gpd
from shapely.geometry import Point, Polygon
import pandas as pd
from omegaconf import OmegaConf
from src.quadnode import QuadNode


def compute_translation(bbox_dict, transform):
    xmin, ymin, zmin = bbox_dict['min_bound']
    xmax, ymax, zmax = bbox_dict['max_bound']
    center = np.vstack([np.array([xmax + xmin, ymax + ymin, zmax + zmin]).reshape((3,1)) / 2, np.array([1])])
    translated = np.linalg.matmul(transform, center)
    norm = float(np.linalg.norm(translated - center))
    norm2d = float(np.linalg.norm(translated[0:2] - center[0:2]))
    direction = (translated[0:2] - center[0:2]) / norm2d 

    return norm, direction


def trim_branch(node):
    if node.children != []:
        for child in node.children:
            trim_branch(child)
            
    # Detach from parent
    if node.parent is not None and node in node.parent.children:
        node.parent.children.remove(node)
        if node.parent.children == []:
            node.parent.is_leaf = True

    # Break all references so pickle can't follow them
    node.parent = None
    node.children = []
    

def detect_absurds(node):
    counter = 0
    norm, direction = compute_translation(node.bbox, node.transform)
    node.metrics['translation_direction'] = direction
    node.metrics['translation_norm'] = norm
    diff = 0

    if node.level > 0:
        parent = node.parent
        parent_norm, _ = compute_translation(parent.bbox, parent.transform)
        diff = abs(parent_norm - norm)

    if diff > 10:
        counter += 1
        for child in node.children:
            trim_branch(child)
        node.metrics['translation_direction'] = parent.metrics['translation_direction']
        node.metrics['translation_norm'] = parent.metrics['translation_norm']
        node.is_absurd = True
        node.is_leaf = True
        for child in node.children:
            child.parent = None
        node.children = []

    for child in node.children:
        counter += detect_absurds(child)
    return counter


def export_bboxes(data, columns, bbox_data, output_path, offset, layer_name, crs="EPSG:2056"):
    """
    Export points and bbox polygons as two layers in a single GPKG.

    Parameters:
        data: list of lists [x, y, lvl, val1, ..., valn]
        columns: column names for data
        bbox_data: list of dicts with keys 'min_bound', 'max_bound' and any attributes
        output_path: path with .gpkg extension
        crs: coordinate reference system (default: Swiss LV95)
    """

    df_points = pd.DataFrame(data, columns=columns)
    rows = []
    for bbox in bbox_data:
        minx, miny = bbox["min_bound"][0] + offset[0], bbox["min_bound"][1] + offset[1]
        maxx, maxy = bbox["max_bound"][0] + offset[0], bbox["max_bound"][1] + offset[1]
        poly = Polygon([
            (minx, miny),
            (maxx, miny),
            (maxx, maxy),
            (minx, maxy),
            (minx, miny)
        ])
        row = {k: v for k, v in bbox.items() if k not in ("min_bound", "max_bound")}
        row["geometry"] = poly
        rows.append(poly)

    gdf_bboxes = gpd.GeoDataFrame(df_points, geometry=rows, crs=crs)
    gdf_bboxes.to_file(output_path, layer=layer_name, driver="GPKG")  # append to same file


def export_points_and_bboxes(data, columns, bbox_data, output_path, offset, crs="EPSG:2056"):
    """
    Export points and bbox polygons as two layers in a single GPKG.

    Parameters:
        data: list of lists [x, y, lvl, val1, ..., valn]
        columns: column names for data
        bbox_data: list of dicts with keys 'min_bound', 'max_bound' and any attributes
        output_path: path with .gpkg extension
        crs: coordinate reference system (default: Swiss LV95)
    """

    # --- Layer 1: Points ---
    # apply offset
    df_points = pd.DataFrame(data, columns=columns)
    geometry_points = [Point(xy) for xy in zip(df_points["x"], df_points["y"])]
    gdf_points = gpd.GeoDataFrame(df_points, geometry=geometry_points, crs=crs)
    gdf_points.to_file(output_path, layer="displacements", driver="GPKG")

    # --- Layer 2: BBoxes ---
    rows = []
    for bbox in bbox_data:
        minx, miny = bbox["min_bound"][0] + offset[0], bbox["min_bound"][1] + offset[1]
        maxx, maxy = bbox["max_bound"][0] + offset[0], bbox["max_bound"][1] + offset[1]
        poly = Polygon([
            (minx, miny),
            (maxx, miny),
            (maxx, maxy),
            (minx, maxy),
            (minx, miny)
        ])
        row = {k: v for k, v in bbox.items() if k not in ("min_bound", "max_bound")}
        row["geometry"] = poly
        rows.append(poly)

    gdf_bboxes = gpd.GeoDataFrame(df_points, geometry=rows, crs=crs)
    gdf_bboxes.to_file(output_path, layer="tiles", driver="GPKG")  # append to same file


def compute_data_for_gpkg(node, data, bbox_data, offset):
    if isinstance(node, QuadNode):
        # compute translation
        bbox_dict = node.bbox
        bbox_data.append(bbox_dict)

        # position:
        xmin, ymin, zmin = bbox_dict['min_bound']
        xmax, ymax, zmax = bbox_dict['max_bound']
        center = np.vstack([np.array([xmax + xmin, ymax + ymin, zmax + zmin]).reshape((3,1)) / 2, np.array([1])])

        data.append(
            (center[0][0] + offset[0], center[1][0] + offset[1], center[2][0] + offset[2], 
            node.metrics['translation_norm'], 
            node.metrics['translation_direction'][0][0], 
            node.metrics['translation_direction'][1][0], 
            node.fitness, node.inlier_rmse, node.planarity,
            node.level, node.is_leaf, node.is_absurd))

        for child in node.children:
            compute_data_for_gpkg(child, data, bbox_data, offset)
    elif isinstance(node, list):
        for el in node:
            # compute translation
            bbox_dict = el.bbox
            bbox_data.append(bbox_dict)

            # position:
            xmin, ymin, zmin = bbox_dict['min_bound']
            xmax, ymax, zmax = bbox_dict['max_bound']
            center = np.vstack([np.array([xmax + xmin, ymax + ymin, zmax + zmin]).reshape((3,1)) / 2, np.array([1])])

            data.append(
                (center[0][0] + offset[0], center[1][0] + offset[1], center[2][0] + offset[2], 
                el.metrics['translation_norm'], 
                el.metrics['translation_direction'][0][0], 
                el.metrics['translation_direction'][1][0], 
                el.fitness, el.inlier_rmse, el.planarity,
                el.level, el.is_leaf, el.is_absurd))
    else:
        raise ValueError("The node need to be a list of Quadtree nodes or the root of a Quadtree")


def tree_to_list(node, list_nodes):
    if node.level > len(list_nodes) - 1:
        list_nodes.append([])
    list_nodes[node.level].append(node)
    for child in node.children:
        tree_to_list(child, list_nodes)


def compute_planarity(points):
    """
    Returns planarity in [0, 1]. Close to 1 = flat plane. Close to 0 = complex geometry.
    Based on eigenvalues of the covariance matrix.
    """
    if len(points) < 3:
        return 1.0
    
    cov = np.cov(points.T)
    eigenvalues = np.sort(np.linalg.eigvalsh(cov))  # ascending: e0 <= e1 <= e2
    e0, e1, e2 = eigenvalues

    total = e0 + e1 + e2 + 1e-10
    planarity = (e1 - e0) / total  # high when e0 ≈ 0 and e1 ≈ e2
    return planarity


def postprocessing(src_transforms, verbose=False):

    # prepare paths
    src_out_absurds = src_transforms.split('.pickle')[0] + "_with_absurd_vals.pickle"
    src_out_gpkg = os.path.join(os.path.dirname(src_transforms), 'points_translate.gpkg')
    src_out_gpkg_leaves = src_out_gpkg.split('.gpkg')[0] + "_leaves.gpkg"
    src_out_gpkg_layers = src_out_gpkg.split('.gpkg')[0] + "_layers.gpkg"
    src_offset = os.path.join(os.path.dirname(src_transforms), 'offset.txt')

    with open(src_transforms, 'rb') as f:
        root = pickle.load(f)
    
    offset = np.loadtxt(src_offset, delimiter=',')

    counter = detect_absurds(root)

    print("Number of absurd values: ", counter)

    with open(src_out_absurds, 'wb') as f:
        pickle.dump(root, f)

    data = []
    bbox_data = []

    compute_data_for_gpkg(root, data, bbox_data, offset)

    columns = ['x', 'y', 'z', 'Disp3D', 'translation_x', 'translation_y', 'fitness', 'inlier_rmse', 'planarity', 'lvl', 'is_leaf', 'absurd_status']

    # Export all tiles
    export_points_and_bboxes(
        data=data,
        bbox_data=bbox_data,
        columns=columns,
        output_path=src_out_gpkg,
        offset=offset,
    )

    # Export only leaves
    mask_leaves = np.array([x[-2] for x in data], dtype=np.bool)
    data_leaves = list(np.array(data)[mask_leaves])
    bbox_data_leaves = list(np.array(bbox_data)[mask_leaves])

    export_points_and_bboxes(
        data=data_leaves,
        bbox_data=bbox_data_leaves,
        columns=columns,
        output_path=src_out_gpkg_leaves,
        offset=offset,
    )

    # Layer by layer
    list_lvls = []
    tree_to_list(root, list_lvls)
    for lvl in range(len(list_lvls)):
        if verbose:
            print("level: ", lvl, ' - num subtiles: ', len(list_lvls[lvl]))

        data = []
        bbox_data = [] 
        compute_data_for_gpkg(list_lvls[lvl], data, bbox_data, offset)

        # Export all tiles
        export_bboxes(
            data=data,
            bbox_data=bbox_data,
            columns=columns,
            output_path=src_out_gpkg_layers,
            offset=offset,
            layer_name=f"Level {lvl}"
        )


if __name__ == "__main__":
    conf = OmegaConf.load('./config.yaml')
    if conf.postprocessing.src_transforms == 'default':
        if conf.data.src_res == 'default':
            conf.data.src_res = os.path.join(os.path.dirname(conf.data.src_pc1), 'results')
        src_transforms = os.path.join(conf.data.src_res, f'pyramid_transforms_{conf.data.res_suffixe}.pickle')
    else:
        src_transforms = conf.postprocessing.src_transforms
    postprocessing(src_transforms)
