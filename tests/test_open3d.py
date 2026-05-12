import os
import numpy as np
import open3d as o3d
import laspy
from time import time
from itertools import product
from tqdm import tqdm
import pickle


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


def compute_transformation(src_pc1, src_pc2):
    # Load files
    source = read_pc(src_pc1)
    target = read_pc(src_pc2)

    # ICP parameters
    max_corr_dist = 0.05  # in the same units as your point cloud
    init_trans = np.eye(4)  # optional initial transformation

    # Run ICP
    reg = o3d.pipelines.registration.registration_icp(
        source,
        target,
        max_corr_dist,
        init_trans,
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )

    return reg.fitness, reg.inlier_rmse, reg.transformation


# def split_pointcloud(points, n):
#     """
#     Recursively split points into 2**n tiles along XY.
#     points: np.ndarray of shape (N, 3)
#     n: number of recursive splits
#     returns: list of np.ndarray, each representing a tile
#     """
#     tiles = [points]
#     for _ in range(n):
#         new_tiles = []
#         for tile in tiles:
#             # Compute mid points
#             xmin, ymin = tile[:, :2].min(axis=0)
#             xmax, ymax = tile[:, :2].max(axis=0)
#             xmid = (xmin + xmax) / 2
#             ymid = (ymin + ymax) / 2

#             # Split into 4 quadrants
#             q1 = tile[(tile[:,0] <= xmid) & (tile[:,1] <= ymid)]
#             q2 = tile[(tile[:,0] > xmid) & (tile[:,1] <= ymid)]
#             q3 = tile[(tile[:,0] <= xmid) & (tile[:,1] > ymid)]
#             q4 = tile[(tile[:,0] > xmid) & (tile[:,1] > ymid)]

#             new_tiles.extend([q1, q2, q3, q4])
#         tiles = new_tiles
#     return tiles


def pyramid_of_transforms(src_pc1, src_pc2, max_level:int):
    temp_bbox = 0
    assert max_level > 0
    max_level += 1  # to process the level without subdivision
    source = read_pc(src_pc1)
    target = read_pc(src_pc2)
    dict_levels = {i: [0, 0, np.zeros((2**i, 2**i, 4, 4))] for i in range(max_level)}

    pointclouds = {
        'source': source,
        'target': target,
    }
    boundaries = {}
    for tile in ['source', 'target']:
        # Get min/max along each axis
        points = np.asarray(pointclouds[tile].points)
        xmin, ymin, zmin = np.astype(points.min(axis=0), np.uint32)
        xmax, ymax, zmax = np.astype(points.max(axis=0), np.uint32)
        boundaries[tile] = [xmin, ymin, xmax, ymax, zmin, zmax]

    for i in range(max_level):
        print("Processing level ", i)
        rangesvals = {}

        for tile in ['source', 'target']:
            xmin, ymin, xmax, ymax, _, _ = boundaries[tile]
            # xranges = [(ix, int(x)) for ix, x in enumerate(np.arange(xmin, xmax, (xmax-xmin)/2**i))]
            # yranges = [(iy, int(y)) for iy, y in enumerate(np.arange(ymin, ymax, (ymax-ymin)/2**i))]
            xranges = [int(x) for x in np.arange(xmin, xmax, (xmax-xmin)/2**i)]
            yranges = [int(y) for y in np.arange(ymin, ymax, (ymax-ymin)/2**i)]
            rangesvals[tile] = {
                'xranges': xranges,
                'yranges': yranges,
            }

        # loop on cells
        combinations = list(product(list(range(2**i)), range(2**i)))
        for _, (ix, iy) in tqdm(enumerate(combinations), total=len(combinations)):
            assert rangesvals['source']['xranges'][ix] == rangesvals['target']['xranges'][ix]
            assert rangesvals['source']['yranges'][iy] == rangesvals['target']['yranges'][iy]

            xmin = rangesvals['source']['xranges'][ix]
            ymin = rangesvals['source']['yranges'][iy]
            xmax = rangesvals['source']['xranges'][ix+1]-1 if ix < 2**i-1 else boundaries['source'][2]
            ymax = rangesvals['source']['yranges'][iy+1]-1 if iy < 2**i-1 else boundaries['source'][3] 

            subtiles = {}
            for tile in ['source', 'target']:
                min_bound = [xmin, ymin, boundaries[tile][-2]]
                max_bound = [xmax, ymax, boundaries[tile][-1]]

                tile_bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
                subtiles[tile] = pointclouds[tile].crop(tile_bbox)
            
            reg = o3d.pipelines.registration.registration_icp(
                subtiles['source'],
                subtiles['target'],
                max_correspondence_distance=0.5,
                init=np.eye(4),
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint()
            )
            fitness = reg.fitness
            dict_levels[i][0] = reg.fitness
            dict_levels[i][1] = reg.inlier_rmse
            dict_levels[i][2] = reg.transformation
    return dict_levels

        



if __name__ == "__main__":
    start = time()
    src_pc1 = 'data/test_pcl/2500_1144_2018.ply'
    src_pc2 = 'data/test_pcl/2500_1144_2024.ply'
    src_result = 'data/test_pcl/2500_1144_transform_pyramid.pickle'
    dict_levels = pyramid_of_transforms(src_pc1, src_pc2, 1)
    with open(src_result, 'wb') as f:
        pickle.dump(dict_levels, f)
    # fitness, rmse, transform = compute_transformation(src_pc1, src_pc2)
    # print('Fitness : ', fitness)
    # print('Root Mean Square Error : ', rmse)
    # print('Transform : ', transform)
    print("Executed in ", round(time() - start, 2), " seconds")