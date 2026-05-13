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
from math import atan2, asin, degrees, sqrt


def remove_A0(node, A0_inv):
    node.global_transform = np.linalg.matmul(node.global_transform, A0_inv)
    for child in node.children:
        remove_A0(child, A0_inv)


def compute_translation(node):
    # Compute translation:
    xmin, ymin, zmin = node.bbox['min_bound']
    xmax, ymax, zmax = node.bbox['max_bound']
    center = np.vstack([np.array([xmax + xmin, ymax + ymin, zmax + zmin]).reshape((3,1)) / 2, np.array([1])])
    translated = np.linalg.matmul(node.global_transform, center)
    diff = translated - center
    norm = float(np.linalg.norm(diff))
    norm2d = float(np.linalg.norm(diff[0:2]))
    direction = ((translated[0:2] - center[0:2]) / norm2d).squeeze(-1)


    node.metrics['pos_i'] = center[:3].squeeze(-1)
    node.metrics['pos_f'] = translated[:3].squeeze(-1)
    node.metrics['translation_x'] = direction[0]
    node.metrics['translation_y'] = direction[1]
    node.metrics['dx'] = diff[0][0]
    node.metrics['dy'] = diff[1][0]
    node.metrics['dz'] = diff[2][0]
    node.metrics['Disp2D'] = norm2d
    node.metrics['Disp3D'] = norm

    for child in node.children:
        compute_translation(child)


def compute_rotation(node):
    # Compute translation at bbox center
    xmin, ymin, zmin = node.bbox['min_bound']
    xmax, ymax, zmax = node.bbox['max_bound']
    center = np.vstack([
        np.array([(xmax + xmin) / 2, 
                  (ymax + ymin) / 2, 
                  (zmax + zmin) / 2]).reshape((3, 1)), 
        np.array([1])
    ])
    translated = np.linalg.matmul(node.global_transform, center)
    diff = translated - center
    dx, dy, dz = list(diff[:3].squeeze(-1))

    norm_3d = float(np.linalg.norm(diff[:3]))

    # Displacement direction: azimuth from North (Y axis), clockwise, in [0, 360°]
    # atan2(east, north) = atan2(dx, dy)
    DispDir = (degrees(atan2(dx, dy)) + 360) % 360

    # Displacement plunge: angle below horizontal, in [0°, 90°]
    # positive = downward
    DispPlunge = degrees(asin(-dz / (norm_3d + 1e-10)))

    # Topple direction: azimuth of the rotation axis projected on horizontal plane
    # Extract rotation axis from the transform
    R = node.global_transform[:3, :3]
    cos_angle = np.clip((np.trace(R) - 1) / 2, -1, 1)
    rotation_angle = degrees(np.arccos(cos_angle))

    # Rotation axis from skew-symmetric part of R
    axis = np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1]
    ])
    axis_norm = np.linalg.norm(axis)
    if axis_norm > 1e-10:
        axis = axis / axis_norm
    else:
        axis = np.array([0, 0, 1])  # no rotation, arbitrary axis

    # Topple direction = azimuth of horizontal projection of rotation axis
    ToppleDir = (degrees(atan2(axis[0], axis[1])) + 360) % 360

    node.metrics['DispDir'] = DispDir
    node.metrics['DispPlunge'] = DispPlunge
    node.metrics['ToppleDir'] = ToppleDir
    node.metrics['rotation_angle'] = rotation_angle
    
    for child in node.children:
        compute_rotation(child)   


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

    # Compute local transform
    xmin, ymin, zmin = node.bbox['min_bound']
    xmax, ymax, zmax = node.bbox['max_bound']
    center = np.vstack([np.array([xmax + xmin, ymax + ymin, zmax + zmin]).reshape((3,1)) / 2, np.array([1])])
    translated = np.linalg.matmul(node.local_transform, center)
    diff = translated - center
    norm_local = float(np.linalg.norm(diff))

    # Apply changes if value absurd
    if norm_local > absurd_th:
        counter += 1
        for child in node.children:
            trim_branch(child)
        for m in node.metrics.keys():
            node.metrics[m] = node.parent.metrics[m]
        node.is_absurd = True
        node.is_leaf = True

        for child in node.children:
            child.parent = None

        node.children = []

    for child in node.children:
        counter += detect_absurds(child, absurd_th)
    return counter


def node_to_list(node, offset=(0,0,0)):
    attributes_name = []
    attributes_val = []
    for key, val in vars(node).items():
        if key in ['children', 'parent', 'global_transform', 'local_transform'] or 'indices' in key:
            continue
        if key == "center":
            attributes_name.append(key)
            attributes_val.append(val)
            for keyc, valc, of in zip(['x', 'y', 'z'], val, offset):
                attributes_name.append(keyc)
                attributes_val.append(valc + of)
        elif key == 'metrics':
            for mkey, mval in val.items():
                if isinstance(mval, np.ndarray):
                    mval = list(mval)
                attributes_name.append(mkey)
                attributes_val.append(mval)
        else:
            attributes_name.append(key)
            attributes_val.append(val)
    return attributes_name, attributes_val


def compute_data_for_gpkg(node, offset):
    data, bbox_data = [], []
    if isinstance(node, QuadNode):
        # compute translation
        bbox_dict = node.bbox
        bbox_data.append([bbox_dict])
        data.append([node_to_list(node, offset)[1]])

        for child in node.children:
            sub_data, sub_bbox_data = compute_data_for_gpkg(child, offset)
            data.append(sub_data)
            bbox_data.append(sub_bbox_data)

        # flattening
        data = [x for row in data for x in row]
        bbox_data = [x for row in bbox_data for x in row]
    elif isinstance(node, list):
        for el in node:
            # compute translation
            bbox_dict = el.bbox
            bbox_data.append(bbox_dict)
            data.append(node_to_list(el, offset)[1])
    else:
        raise ValueError("The node need to be a list of Quadtree nodes or the root of a Quadtree")
    
    return data, bbox_data


def export_points_and_bboxes(data, columns, bbox_data, output_path, offset, to_export='both', layer_name='', crs="EPSG:2056"):
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
    # --- Layer 1: Points ---
    if to_export in ['points', 'both']:
        layer_name_sub = f"{layer_name}_centers" if layer_name != '' else 'centers'
        geometry_points = [Point(xy) for xy in zip(df_points["x"], df_points["y"])]
        gdf_points = gpd.GeoDataFrame(df_points, geometry=geometry_points, crs=crs)
        gdf_points.to_file(output_path, layer=layer_name_sub, driver="GPKG")

    # --- Layer 2: BBoxes ---
    if to_export in ['boxes', 'both']:
        layer_name_sub = f"{layer_name}_tiles" if layer_name != '' else 'tiles'
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
        gdf_bboxes.to_file(output_path, layer=layer_name_sub, driver="GPKG")  # append to same file


def tree_to_list(node, list_tot, list_per_level):
    list_tot.append(node)
    if node.level > len(list_per_level) - 1:
        list_per_level.append([])
    list_per_level[node.level].append(node)
    for child in node.children:
        tree_to_list(child, list_tot, list_per_level)


def postprocessing(root, src_out_gpkg, offset, absurd_dist=5, suffixe='', verbose=False):

    # prepare paths
    src_out_gpkg = src_out_gpkg.split('.gpkg')[0] + f"_{suffixe}.gpkg"
    src_out_gpkg_leaves = src_out_gpkg.split('.gpkg')[0] + f"_leaves.gpkg"
    src_out_gpkg_layers_tiles = src_out_gpkg.split('.gpkg')[0] + f"_layers_tiles.gpkg"
    src_out_gpkg_layers_centers = src_out_gpkg.split('.gpkg')[0] + f"_layers_centers.gpkg"

    # store nodes in a list
    list_nodes, list_nodes_per_level = [], []
    tree_to_list(root, list_nodes, list_nodes_per_level)

    # Compute metrics:
    compute_translation(root)
    compute_rotation(root)

    # Detect absurd values
    original_len = len(root)
    counter = detect_absurds(root, absurd_dist)
    print(f"Number of absurd values: {counter} ({np.round(counter/original_len*100, 2)}%)")

    # Compute coherence indexes
    translation_x = [node.metrics['translation_x'] for node in list_nodes]
    translation_y = [node.metrics['translation_y'] for node in list_nodes]
    displacement = [node.metrics['Disp3D'] for node in list_nodes]
    planarity = [float(node.planarity) for node in list_nodes]
    rotation_angles = [node.metrics['rotation_angle'] for node in list_nodes]

    spatial_coherences = compute_spatial_coherence(list_nodes, translation_x, translation_y, 40)
    magnitude_zscores = compute_magnitude_zscore(list_nodes, displacement, 40)

    mask_artifact = (
        (np.array(spatial_coherences) < 0.707) |       # direction disagrees with neighbors (> 45°)
        (np.array(magnitude_zscores) > 2.5) |           # magnitude is outlier among neighbors
        (np.array(rotation_angles) > 5) |              # large rotation
        (np.array(planarity) > 0.999)           # flat surface → degenerate ICP
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
    data, bbox_data = compute_data_for_gpkg(root, offset)

    columns = node_to_list(root)[0]

    # Export all tiles
    export_points_and_bboxes(
        data=data,
        bbox_data=bbox_data,
        columns=columns,
        output_path=src_out_gpkg,
        offset=offset,
    )

    # Export only leaves
    data_leaves = [x for x in data if x[-2] == True]
    mask_leaves = np.array([x[-2] for x in data], dtype=np.bool)
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

        data, bbox_data = compute_data_for_gpkg(list_nodes_per_level[lvl], offset)

        export_points_and_bboxes(
            data=data,
            bbox_data=bbox_data,
            columns=columns,
            output_path=src_out_gpkg_layers_tiles,
            to_export='boxes',
            offset=offset,
            layer_name=f"Level {lvl}"
        )

        export_points_and_bboxes(
            data=data,
            bbox_data=bbox_data,
            columns=columns,
            output_path=src_out_gpkg_layers_centers,
            to_export='points',
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

    # prepare paths
    src_out_gpkg = os.path.join(os.path.dirname(src_transforms), 'points_translate.gpkg')
    src_offset = os.path.join(os.path.dirname(src_transforms), 'offset.txt')

    with open(src_transforms, 'rb') as f:
        root = pickle.load(f)
    offset = np.loadtxt(src_offset, delimiter=',')

    absurd_th = float(conf.postprocessing.absurd_dist)
    # Postprocess with A0
    print("Postprocessing with initial alignment (w_A0)")
    postprocessing(root, src_out_gpkg, offset, absurd_th, 'w_A0')

    # Postprocess without A0:
    print("\nPostprocessing without initial alignment (wo_A0)")
    A0_inv = np.linalg.inv(root.global_transform)
    remove_A0(root, A0_inv)
    postprocessing(root, src_out_gpkg, offset, absurd_th, 'wo_A0')
    print()