# dependencies
import os
import numpy as np
import pickle
import geopandas as gpd
from shapely.geometry import Point, Polygon
import pandas as pd
from omegaconf import OmegaConf
from src.quadnode import QuadNode
from src.coherence import compute_spatial_coherence, compute_magnitude_zscore, compute_rotation_angles, compute_confidence


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
    

def detect_absurds(node, absurd_th):
    counter = 0
    tot = 0
    norm, direction = compute_translation(node.bbox, node.transform)
    node.metrics['translation_direction'] = direction
    node.metrics['Disp3D'] = norm
    diff = 0

    if node.level > 0:
        parent = node.parent
        parent_norm, _ = compute_translation(parent.bbox, parent.transform)
        diff = abs(parent_norm - norm)

    if diff > absurd_th:
        counter += 1
        for child in node.children:
            trim_branch(child)
        node.metrics['translation_direction'] = parent.metrics['translation_direction']
        node.metrics['Disp3D'] = parent.metrics['Disp3D']
        node.is_absurd = True
        node.is_leaf = True
        for child in node.children:
            child.parent = None
        node.children = []

    for child in node.children:
        counter += detect_absurds(child, absurd_th)
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
            node.metrics['Disp3D'], 
            node.metrics['translation_direction'][0][0], 
            node.metrics['translation_direction'][1][0], 
            node.fitness, node.inlier_rmse, node.planarity,
            node.metrics['spatial_coherence'],
            node.metrics['magnitude_zscore'],
            node.metrics['rotation_angle'],
            node.metrics['confidence'],
            node.metrics['is_artifact'],
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
                el.metrics['Disp3D'], 
                el.metrics['translation_direction'][0][0], 
                el.metrics['translation_direction'][1][0], 
                el.fitness, el.inlier_rmse, el.planarity,
                el.metrics['spatial_coherence'],
                el.metrics['magnitude_zscore'],
                el.metrics['rotation_angle'],
                el.metrics['confidence'],
                el.metrics['is_artifact'],
                el.level, el.is_leaf, el.is_absurd))
    else:
        raise ValueError("The node need to be a list of Quadtree nodes or the root of a Quadtree")


def tree_to_list(node, list_tot, list_per_level):
    list_tot.append(node)
    if node.level > len(list_per_level) - 1:
        list_per_level.append([])
    list_per_level[node.level].append(node)
    for child in node.children:
        tree_to_list(child, list_tot, list_per_level)


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


    # Detect absurd values
    original_len = len(root)
    counter = detect_absurds(root, 5)
    print(f"Number of absurd values: {counter} ({np.round(counter/original_len*100, 2)}%)")

    list_nodes, list_nodes_per_level = [], []
    tree_to_list(root, list_nodes, list_nodes_per_level)
    # list_leaves = [x for x in list_nodes if x.is_leaf]

    with open(src_out_absurds, 'wb') as f:
        pickle.dump(root, f)

    # Compute coherence indexes
    translation_x = [node.metrics['translation_direction'][0][0] for node in list_nodes]
    translation_y = [node.metrics['translation_direction'][1][0] for node in list_nodes]
    displacement = [node.metrics['Disp3D'] for node in list_nodes]
    planarity = [node.planarity for node in list_nodes]
    transforms = [node.transform for node in list_nodes]

    spatial_coherences = compute_spatial_coherence(list_nodes, translation_x, translation_y, 40)
    magnitude_zscores = compute_magnitude_zscore(list_nodes, displacement, 40)
    rotation_angles = compute_rotation_angles(transforms)

    mask_artifact = (
        (np.array(spatial_coherences) < 0.707) |       # direction disagrees with neighbors (> 45°)
        (np.array(magnitude_zscores) > 2.5) |           # magnitude is outlier among neighbors
        (np.array(rotation_angles) > 5) |              # large rotation
        (np.array(planarity) > 0.485)           # flat surface → degenerate ICP
    )
    print(f"Number of masked samples: {np.sum(mask_artifact)} ({np.round(np.sum(mask_artifact)/mask_artifact.shape[0]*100, 2)}%)")

    for node, coherence, magnitude, rotation, artifact in zip(list_nodes, spatial_coherences, magnitude_zscores, rotation_angles, mask_artifact):
        node.metrics['spatial_coherence'] = coherence
        node.metrics['magnitude_zscore'] = magnitude
        node.metrics['rotation_angle'] = rotation
        node.metrics['confidence'] = compute_confidence(coherence, magnitude, rotation, node.planarity, node.fitness, node.inlier_rmse, 
                                                        w_fitness=0, w_rmse=0)
        node.metrics['is_artifact'] = bool(artifact)

    # Gather data for GPKG
    data = []
    bbox_data = []

    compute_data_for_gpkg(root, data, bbox_data, offset)

    columns = ['x', 'y', 'z', 'Disp3D', 'translation_x', 'translation_y', 'fitness', 'inlier_rmse', 
               'planarity', 'spatial_coherence', 'magnitude_zscore', 'rotation_angle', 'confidence',
               'is_artifact', 'lvl', 'is_leaf', 'absurd_status']

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
    for lvl in range(len(list_nodes_per_level)):
        if verbose:
            print("level: ", lvl, ' - num subtiles: ', len(list_nodes_per_level[lvl]))

        data = []
        bbox_data = [] 
        compute_data_for_gpkg(list_nodes_per_level[lvl], data, bbox_data, offset)

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
