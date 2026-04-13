import os
import numpy as np
import open3d as o3d
import laspy
from time import time
import pickle
from omegaconf import OmegaConf
from copy import deepcopy


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
    

def rec_tile_process(tiles, bbox, lvl, src_res, args, transform):
    source_transformed = deepcopy(tiles['source'])
    if transform is not None:
        source_transformed.transform(transform)

    new_tiles = {
        'source': source_transformed.crop(bbox),
        'target': tiles['target'].crop(bbox),
    }

    # Load method
    method = None
    if args.method == 'pointtopoint':
        method = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    elif args.method == 'pointtoplane':
        method = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    else:
        raise ValueError(f"The given method is wrong!\n\tGiven: {args.method}\n\tAccepted: [pointtopoint, pointtoplane]")
    # method = o3d.pipelines.registration.TransformationEstimationForGeneralizedICP()

    # max_correspondence = 0.5 if lvl == 0 else 1.0/1.1**lvl
    max_correspondence = [0.5, 5, 4, 3, 2, 1, 0.5]
    max_correspondence = [0.5, 5, 4, 3, 2, 0.5]     # res = 0.11    0.035   5   7   12  70
    max_correspondence = [0.5, 5, 4, 3, 1.5, 0.4, 0.4]   # res = 0.11    0.10    7   8   12  50 25000
    # max_correspondence = [0.5, 5, 4, 3, 1.3, 0.3]   # res = 0.11    0.10    7   8   20  500
    # max_correspondence = [0.5, 5, 4, 3, 1, 0.35]    # res = 0.11    0.035   5   7   12  140
    # max_correspondence = [0.5, 5, 4, 3, 1, 0.4]     # res = 0.11    0.035   5   7   12  60
    # max_correspondence = [0.5, 5, 4, 3, 1.5, 0.2]   # res = 0.11    0.035   4   6   14  80  generalised icp
    # max_correspondence = [0.5, 5, 4, 3, 1.5, 0.15]   # res = 0.11    0.035   4   6   14  70  generalised icp
    # max_correspondence = [0.5, 5, 4, 3, 1.5, 0.1]   # res = 0.11    0.035   4   6   14  175  generalised icp
    # print(max_correspondence[lvl])
    reg = o3d.pipelines.registration.registration_icp(
    # reg = o3d.pipelines.registration.registration_generalized_icp(
        new_tiles['source'],
        new_tiles['target'],
        # max_correspondence_distance=0.2,
        # max_correspondence_distance=2,
        # max_correspondence_distance=0.5/1.5**lvl,
        max_correspondence_distance=max_correspondence[lvl],
        # max_correspondence_distance=max_correspondence,
        init=transform if transform is not None else np.eye(4),
        # init=np.eye(4),
        estimation_method=method,
        # criteria=o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1.000000e-03, relative_rmse=1.000000e-03, max_iteration=50)
    )

    # new_transform = deepcopy(reg.transformation)

    # # ignore z transform
    # reg.transformation[2,3] = 0

    # # clip z transform
    # new_transform[2, 3] = np.clip(new_transform[2, 3], -1.0, 1.0)

    # # clip full transform
    # t = np.sqrt(np.sum(new_transform[0:3,3]**2))
    # if t > 10:
    #     new_transform = np.array([[1,0,0,1],[0,1,0,1],[0,0,1,1],[0,0,0,1]])

    # if reg.inlier_rmse > 0.28:
    #     new_transform = np.array([[1,0,0,1],[0,1,0,1],[0,0,1,1],[0,0,0,1]])

    # new_tiles['source'].transform(reg.transformation)
    # new_tiles['target'].transform(reg.transformation)
    # new_tiles['target'].transform(np.linalg.inv(reg.transformation))
    # if isinstance(transform, np.ndarray):
    #     reg.transformation = np.linalg.matmul(transform, reg.transformation)

    # save transformed tile if wanted:
    if args.do_output_transformed and args.output_level in [-1, lvl]:
        x,y,_ = bbox.get_min_bound()
        src_file = os.path.join(src_res, f'alligned_pc_lvl={lvl}_x={x}_y={y}_source.ply')
        o3d.io.write_point_cloud(src_file, new_tiles['source'])
        src_file = os.path.join(src_res, f'alligned_pc_lvl={lvl}_x={x}_y={y}_target.ply')
        o3d.io.write_point_cloud(src_file, new_tiles['target'])



    # ---- TEMP FOR TESTING ---
    x,y,_ = bbox.get_min_bound()
    if lvl == 4 and y == 342.0 and x == 0.0:
        # print(np.linalg.norm(reg.transformation[0:3, 3]))
        src_file = r"D:\GitHubProjects\Terranum_repo\pc_movement_tracking\data\test_real_movement\results\test\original.ply"
        o3d.io.write_point_cloud(src_file, new_tiles['source'])

        src_file = r"D:\GitHubProjects\Terranum_repo\pc_movement_tracking\data\test_real_movement\results\test\target.ply"
        o3d.io.write_point_cloud(src_file, new_tiles['target'])

        transformed = deepcopy(new_tiles['source']).transform(np.linalg.matmul(transform, reg.transformation))
        src_file = r"D:\GitHubProjects\Terranum_repo\pc_movement_tracking\data\test_real_movement\results\test\transformed.ply"
        o3d.io.write_point_cloud(src_file, transformed)
    # -------------------------



    new_transform = np.eye(4) if transform is None else np.linalg.matmul(transform, reg.transformation)
    list_res = []
    if lvl < args.max_level and len(new_tiles['source'].points) > args.min_points:
        list_res.append((lvl, (bbox.get_min_bound(), reg.fitness, reg.inlier_rmse, reg.transformation), False))
        bboxes = compute_bbox(bbox)
        for subbbox in bboxes:
            sublist = rec_tile_process(
                new_tiles, 
                subbbox, 
                lvl+1, 
                src_res, 
                args, 
                # reg.transformation
                new_transform,
                )
            for el in sublist:
                list_res.append(el)
    else:
        # print('Cutting at level: ', lvl)
        list_res.append((lvl, (bbox.get_min_bound(), reg.fitness, reg.inlier_rmse, reg.transformation), True))

    return list_res


def rec_tile_process_full_tile(tiles, bbox, lvl, src_res, args, transform):
    local_tile = deepcopy(tiles['source']).transform(transform)

    # Load method
    method = None
    if args.method == 'pointtopoint':
        method = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    elif args.method == 'pointtoplane':
        method = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    else:
        raise ValueError(f"The given method is wrong!\n\tGiven: {args.method}\n\tAccepted: [pointtopoint, pointtoplane]")

    max_correspondence = [0.5, 5, 4, 3, 1.5, 0.4, 0.4]   # res = 0.11    0.10    7   8   12  50 25000
    
    reg = o3d.pipelines.registration.registration_icp(
    # reg = o3d.pipelines.registration.registration_generalized_icp(
        local_tile.crop(bbox),
        tiles['target'].crop(bbox),
        max_correspondence_distance=max_correspondence[lvl],
        # max_correspondence_distance=2,
        # init=transform if transform is not None else np.eye(4),
        init=np.eye(4),
        estimation_method=method,
    )

    # save transformed tile if wanted:
    if args.do_output_transformed and args.output_level in [-1, lvl]:
        x,y,_ = bbox.get_min_bound()
        src_file = os.path.join(src_res, f'alligned_pc_lvl={lvl}_x={x}_y={y}_source.ply')
        o3d.io.write_point_cloud(src_file, tiles['source'].crop(bbox))
        src_file = os.path.join(src_res, f'alligned_pc_lvl={lvl}_x={x}_y={y}_transformed.ply')
        o3d.io.write_point_cloud(src_file, local_tile.crop(bbox))
        src_file = os.path.join(src_res, f'alligned_pc_lvl={lvl}_x={x}_y={y}_target.ply')
        o3d.io.write_point_cloud(src_file, tiles['target'].crop(bbox))



    # # ---- TEMP FOR TESTING ---
    # x,y,_ = bbox.get_min_bound()
    # if lvl == 4 and y == 342.0 and x == 0.0:
    #     # print(np.linalg.norm(reg.transformation[0:3, 3]))
    #     src_file = r"D:\GitHubProjects\Terranum_repo\pc_movement_tracking\data\test_real_movement\results\test_full\original.ply"
    #     o3d.io.write_point_cloud(src_file, new_tiles['source'])

    #     src_file = r"D:\GitHubProjects\Terranum_repo\pc_movement_tracking\data\test_real_movement\results\test_full\target.ply"
    #     o3d.io.write_point_cloud(src_file, new_tiles['target'])

    #     transformed = deepcopy(new_tiles['source']).transform(np.linalg.matmul(transform, reg.transformation))
    #     src_file = r"D:\GitHubProjects\Terranum_repo\pc_movement_tracking\data\test_real_movement\results\test_full\transformed.ply"
    #     o3d.io.write_point_cloud(src_file, transformed)
    # # -------------------------



    new_transform = np.linalg.matmul(transform, reg.transformation)
    list_res = []
    bbox_dict = {
        "min_bound": bbox.get_min_bound().tolist(),
        "max_bound": bbox.get_max_bound().tolist()
    }
    if lvl < args.max_level and len(local_tile.crop(bbox).points) > args.min_points:
        # list_res.append((lvl, (bbox.get_min_bound(), reg.fitness, reg.inlier_rmse, reg.transformation), False))
        # list_res.append((lvl, (bbox_dict, reg.fitness, reg.inlier_rmse, reg.transformation), False))
        list_res.append((lvl, (bbox_dict, reg.fitness, reg.inlier_rmse, new_transform), False))
        bboxes = compute_bbox(bbox)
        for subbbox in bboxes:
            sublist = rec_tile_process_full_tile(
                tiles, 
                subbbox, 
                lvl+1, 
                src_res, 
                args, 
                # reg.transformation
                new_transform,
                )
            for el in sublist:
                list_res.append(el)
    else:
        # print('Cutting at level: ', lvl)
        # list_res.append((lvl, (bbox.get_min_bound(), reg.fitness, reg.inlier_rmse, reg.transformation), True))
        # list_res.append((lvl, (bbox_dict, reg.fitness, reg.inlier_rmse, reg.transformation), True))
        list_res.append((lvl, (bbox_dict, reg.fitness, reg.inlier_rmse, new_transform), True))

    return list_res


if __name__ == "__main__":
    conf = OmegaConf.load("config.yaml")

    # prepare results
    os.makedirs(conf.data.src_res, exist_ok=True)
    pointcloud_res = os.path.join(conf.data.src_res, 'pointclouds')
    if conf.args.do_output_transformed:
        os.makedirs(pointcloud_res, exist_ok=True)

    start = time()
    src_result = os.path.join(conf.data.src_res, f'pyramid_transforms_{conf.data.res_suffixe}.pickle')

    tiles = {
        'source': read_pc(conf.data.src_pc1),
        'target': read_pc(conf.data.src_pc2),
    }

    # Center pointclouds
    z_mean = tiles['source'].compute_mean_and_covariance()[0][2]
    # print(z_mean)
    # print(np.array([-x for x in conf.args.huge_translation]) - np.array([0, 0, z_mean]))
    # quit()
    for tile in tiles.values():
        # z_mean = tile.compute_mean_and_covariance()[0][2]
        tile.translate(np.array([-x for x in conf.args.huge_translation]) - np.array([0, 0, z_mean]))
        # tile.translate(np.array([-x for x in conf.args.huge_translation]))

    # weights = np.array([1, 1, 0.1])
    
    # weights = np.array([[1,0,0,1],[0,1,0,1],[0,0,1,0.1],[0,0,0,1]])
    # scale = np.array([1, 1, 0.1])
    # for tile in tiles.values():
    #     tile.transform(scale)

    #     points = np.asarray(tile.points)
    #     points *= scale
    
    # Compute normals
    if conf.args.method == 'pointtoplane':
        tiles['target'].estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=conf.args.pointtoplane_radius, 
                max_nn=conf.args.pointtoplane_max_nn,
                ))

    # tiles['target'].orient_normals_consistent_tangent_plane(30)

    bbox_source = tiles['source'].get_axis_aligned_bounding_box()
    bbox_target = tiles['target'].get_axis_aligned_bounding_box()

    # list_transforms = rec_tile_process(
    list_transforms = rec_tile_process_full_tile(
        tiles=tiles, 
        bbox=bbox_source, 
        lvl=0, 
        src_res=pointcloud_res, 
        args=conf.args,
        transform=np.eye(4)
        )
    
    with open(src_result, 'wb') as f:
        pickle.dump(list_transforms, f)

    print("Executed in ", round(time() - start, 2), " seconds")