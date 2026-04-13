import os
import numpy as np
import open3d as o3d
import laspy
from time import time
import pickle


def compute_bbox(boundaries):
    min_bound, max_bound = boundaries.get_min_bound(), boundaries.get_max_bound()
    minx, miny, minz = min_bound
    maxx, maxy, maxz = max_bound
    spanx = (maxx-minx) / 2
    spany = (maxy-miny) / 2
    bboxes = []
    for i in range(2):
        for j in range(2):
            x0 = int(minx + i * spanx)
            y0 = int(miny + j * spany)
            x1 = int(x0 + spanx)
            y1 = int(y0 + spany)
            bboxes.append(o3d.geometry.AxisAlignedBoundingBox((x0, y0, minz), (x1, y1, maxz)))
    return bboxes


def read_pc(src_pc):
    ext = os.path.splitext(src_pc)[1].lower()

    if ext in ['.ply', '.pcd', '.xyz', '.xyzrgb', '.xyzn', '.pts']:
        return o3d.io.read_point_cloud(src_pc)
    elif ext in ['.las', '.laz']:
        xyz = laspy.read(src_pc)
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(xyz.xyz)
        return pc
    else:
        raise AttributeError(f"Wrong extension '{ext}'. Should be in '.ply', '.pcd', '.xyz', '.xyzrgb', '.xyzn', '.pts', '.las' and '.laz")
    

def rec_tile_process(tiles, bbox, lvl, lvl_max):
    new_tiles = {
        'source': tiles['source'].crop(bbox),
        'target': tiles['target'].crop(bbox),
    }
    reg = o3d.pipelines.registration.registration_icp(
        new_tiles['source'],
        new_tiles['target'],
        max_correspondence_distance=0.5,
        init=np.eye(4),
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )
    new_tiles['source'].transform(reg.transformation)
    list_res = [(lvl, (bbox.get_min_bound(), reg.fitness, reg.inlier_rmse, reg.transformation))]
    if lvl < lvl_max:
        bboxes = compute_bbox(bbox)
        for subbbox in bboxes:
            sublist = rec_tile_process(new_tiles, subbbox, lvl+1, lvl_max)
            for el in sublist:
                list_res.append(el)

    return list_res


if __name__ == "__main__":
    start = time()
    max_level = 6
    src_pc1 = r"D:\GitHubProjects\Terranum_repo\pc_movement_tracking\data\test_real_movement\PC2015_2589_1169.ply"
    src_pc2 = r"D:\GitHubProjects\Terranum_repo\pc_movement_tracking\data\test_real_movement\PC2019_2589_1169.ply"
    src_result = "D:\GitHubProjects\Terranum_repo\pc_movement_tracking\data\test_real_movement\results_no_quad\pyramid_transforms_test.pickle"

    tiles = {
        'source': read_pc(src_pc1),
        'target': read_pc(src_pc2),
    }
    bbox_source = tiles['source'].get_axis_aligned_bounding_box()
    bbox_target = tiles['target'].get_axis_aligned_bounding_box()

    list_transforms = rec_tile_process(tiles, bbox_source, 0, max_level)
    with open(src_result, 'wb') as f:
        pickle.dump(list_transforms, f)
    print("Executed in ", round(time() - start, 2), " seconds")