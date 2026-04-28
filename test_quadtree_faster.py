import os
import numpy as np
import open3d as o3d
import laspy
from time import time
import pickle
from omegaconf import OmegaConf


class QuadNode:
    """Quadtree node storing spatial bbox, point indices, level, and children."""
    def __init__(self, bbox, indices_src, indices_tgt,indices_with_neigh, indices_sub_pts, level, parent):
        self.bbox = bbox
        self.indices_src = indices_src
        self.indices_with_neigh = indices_with_neigh
        self.indices_sub_pts = indices_sub_pts
        self.indices_tgt = indices_tgt
        self.level = level
        self.parent = parent
        self.children = []
        self.is_leaf = True


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


def read_ply_with_scalars(ply_path):
    """
    Read a PLY file and return an Open3D point cloud + a dict of scalar fields.
    """
    from plyfile import PlyData
    
    plydata = PlyData.read(ply_path)
    vertex = plydata['vertex']
    
    # Extract xyz
    x = np.array(vertex['x'])
    y = np.array(vertex['y'])
    z = np.array(vertex['z'])
    xyz = np.stack([x, y, z], axis=1)
    
    # Build Open3D point cloud
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(xyz)
    
    # Extract colors if available
    if all(c in vertex.data.dtype.names for c in ('red', 'green', 'blue')):
        r = np.array(vertex['red'], dtype=np.float64) / 255.0
        g = np.array(vertex['green'], dtype=np.float64) / 255.0
        b = np.array(vertex['blue'], dtype=np.float64) / 255.0
        pc.colors = o3d.utility.Vector3dVector(np.stack([r, g, b], axis=1))
    
    # Extract all scalar fields (scalar_* prefix)
    scalars = {}
    for name in vertex.data.dtype.names:
        if name.startswith('scalar_'):
            key = name[len('scalar_'):]  # strip prefix for cleaner access
            scalars[key] = np.array(vertex[name])
    
    return pc, scalars


def compute_bbox(boundaries):
    min_bound, max_bound = boundaries.get_min_bound(), boundaries.get_max_bound()
    minx, miny, minz = min_bound
    maxx, maxy, maxz = max_bound
    spanx = (maxx-minx) / 2
    spany = (maxy-miny) / 2
    bboxes = []
    for i in range(2):
        for j in range(2):
            x0 = minx + i * spanx
            y0 = miny + j * spany
            x1 = x0 + spanx
            y1 = y0 + spany
            bboxes.append(o3d.geometry.AxisAlignedBoundingBox((x0, y0, minz), (x1, y1, maxz)))
    return bboxes


def points_in_bbox(xyz_src, xyz_tgt, parent, bbox, indices_src, indices_tgt):
    """Return indices of points inside bbox. xyz: Nx3 array, indices: subset indices."""
    min_b = bbox.get_min_bound()
    max_b = bbox.get_max_bound()
    span = max_b - min_b
    min_b_w_neigh = min_b - span
    max_b_w_neigh = max_b + span

    pts_src = xyz_src[indices_src]
    pts_parent = xyz_src[parent.indices_src] if parent != None else xyz_src
    pts_tgt = xyz_tgt[indices_tgt]

    # compute points in source
    mask = (
        (pts_src[:, 0] >= min_b[0]) & (pts_src[:, 0] < max_b[0]) &
        (pts_src[:, 1] >= min_b[1]) & (pts_src[:, 1] < max_b[1])
    )

    mask_w_neigh = (
        (pts_parent[:, 0] >= min_b_w_neigh[0]) & (pts_parent[:, 0] < max_b_w_neigh[0]) &
        (pts_parent[:, 1] >= min_b_w_neigh[1]) & (pts_parent[:, 1] < max_b_w_neigh[1])
    )

    # sub_pts_src = xyz_src[mask_w_neigh]
    sub_pts_src = pts_parent[mask_w_neigh]
    indices_sub = np.arange(len(sub_pts_src))
    mask_sub = (
        (sub_pts_src[:, 0] >= min_b[0]) & (sub_pts_src[:, 0] < max_b[0]) &
        (sub_pts_src[:, 1] >= min_b[1]) & (sub_pts_src[:, 1] < max_b[1])
    )

    # compute points in target
    mask_tgt = (
        (pts_tgt[:, 0] >= min_b[0]) & (pts_tgt[:, 0] < max_b[0]) &
        (pts_tgt[:, 1] >= min_b[1]) & (pts_tgt[:, 1] < max_b[1])
    )

    indices_w_neigh = parent.indices_src[mask_w_neigh] if parent != None else np.arange(len(xyz_src))[mask_w_neigh]

    return indices_src[mask], indices_tgt[mask_tgt], indices_w_neigh, indices_sub[mask_sub]


# def build_quadtree(xyz_src, xyz_tgt, parent, bbox, indices_src, indices_tgt, indices_with_neigh, indices_sub_pts, level, max_level, min_points):
#     """Recursively build quadtree based on point density."""

#     node = QuadNode(bbox, indices_src, indices_tgt, indices_with_neigh, indices_sub_pts, level, parent)

#     # stopping condition
#     if level == max_level:
#         return node

#     sub_bboxes = compute_bbox(bbox)
    
    
#     children_indices = []
#     for subbbox in sub_bboxes:
#         sub_idx_src, sub_idx_tgt, sub_idx_with_neigh, sub_idx_sub_pts = points_in_bbox(xyz_src, xyz_tgt, parent, subbbox, indices_src, indices_tgt)

#         if len(sub_idx_src) <= min_points or len(sub_idx_tgt) <= min_points:
#             continue

#         children_indices.append((subbbox, sub_idx_src, sub_idx_tgt, sub_idx_with_neigh, sub_idx_sub_pts))

#     # stopping condition
#     if len(children_indices) != 4:
#         return node
    
#     for (subbbox, sub_idx_src, sub_idx_tgt, sub_idx_with_neigh, sub_idx_sub_pts) in children_indices:
#         child = build_quadtree(
#             xyz_src, xyz_tgt, node, subbbox, sub_idx_src, sub_idx_tgt, sub_idx_with_neigh, sub_idx_sub_pts,
#             level + 1, max_level, min_points
#         )
#         node.children.append(child)

#     if node.children:
#         node.is_leaf = False

#     return node


def build_quadtree(xyz_src, xyz_tgt, parent, bbox, indices_src, indices_tgt, indices_with_neigh, indices_sub_pts, level, max_level, min_points):
    """Recursively build quadtree based on point density."""

    node = QuadNode(bbox, indices_src, indices_tgt, indices_with_neigh, indices_sub_pts, level, parent)
    # print(bbox.get_min_bound())
    # print(bbox.get_max_bound())
    # print('---')
    # stopping condition
    if level >= max_level or len(indices_src) <= min_points or len(indices_tgt) <= min_points:
        return node

    sub_bboxes = compute_bbox(bbox)

    for subbbox in sub_bboxes:
        sub_idx_src, sub_idx_tgt, sub_idx_with_neigh, sub_idx_sub_pts = points_in_bbox(xyz_src, xyz_tgt, parent, subbbox, indices_src, indices_tgt)

        if len(sub_idx_src) == 0 or len(sub_idx_tgt) == 0:
            continue

        child = build_quadtree(
            xyz_src, xyz_tgt, node, subbbox, sub_idx_src, sub_idx_tgt, sub_idx_with_neigh, sub_idx_sub_pts,
            level + 1, max_level, min_points
        )
        node.children.append(child)

    if node.children:
        node.is_leaf = False

    return node


# def build_quadtree(xyz_src, xyz_tgt, bbox, indices_src, indices_tgt, indices_with_neigh, indices_sub_pts, level, max_level, min_points):
#     """Recursively build quadtree based on point density."""

#     node = QuadNode(bbox, indices_src, indices_tgt, indices_with_neigh, indices_sub_pts, level)

#     # stopping condition
#     if level >= max_level or len(indices_src) <= min_points or len(indices_tgt) <= min_points:
#         return node

#     sub_bboxes = compute_bbox(bbox)

#     for subbbox in sub_bboxes:
#         sub_idx_src, sub_idx_tgt, sub_idx_with_neigh, sub_idx_sub_pts = points_in_bbox(xyz_src, xyz_tgt, subbbox, indices_src, indices_tgt)

#         if len(sub_idx_src) == 0 or len(sub_idx_tgt) == 0:
#             continue

#         child = build_quadtree(
#             xyz_src, xyz_tgt, subbbox, sub_idx_src, sub_idx_tgt, sub_idx_with_neigh, sub_idx_sub_pts,
#             level + 1, max_level, min_points
#         )
#         node.children.append(child)

#     if node.children:
#         node.is_leaf = False

#     return node


def extract_subcloud(pc, indices):
    """Return a sub pointcloud from indices (including normals if available)."""
    
    sub_pc = o3d.geometry.PointCloud()

    pts = np.asarray(pc.points)
    sub_pc.points = o3d.utility.Vector3dVector(pts[indices])

    # Copy normals if they exist
    if pc.has_normals():
        normals = np.asarray(pc.normals)
        sub_pc.normals = o3d.utility.Vector3dVector(normals[indices])

    return sub_pc


def run_icp_on_tree(node, pc_source, pc_target, src_res, args, transform, results, time_subclouds_creation, time_icp, time_transform):
    """Traverse tree and run ICP on each node."""
    
    x,y,_ = node.bbox.get_min_bound()
    time_sub_0 = time()
    src_sub_w_neigh = extract_subcloud(pc_source, node.indices_with_neigh)
    time_subclouds_creation_local = time() - time_sub_0

    # save transformed tile if wanted:
    if args.do_output_transformed and args.output_level in [-1, node.level]:
        src_file = os.path.join(src_res, f'alligned_pc_lvl={node.level}_x={x}_y={y}_source.ply')
        o3d.io.write_point_cloud(src_file, src_sub_w_neigh)

    # transform local sample
    time_sub_0 = time()
    src_sub_w_neigh.transform(transform)
    time_transform.append(time() - time_sub_0)

    time_sub_0 = time()
    src_sub = extract_subcloud(src_sub_w_neigh, node.indices_sub_pts)
    tgt_sub = extract_subcloud(pc_target, node.indices_tgt)
    time_subclouds_creation_local += time() - time_sub_0
    time_subclouds_creation.append(time_subclouds_creation_local)

    # save transformed tile if wanted:
    if args.do_output_transformed and args.output_level in [-1, node.level]:
        src_file = os.path.join(src_res, f'alligned_pc_lvl={node.level}_x={x}_y={y}_target.ply')
        o3d.io.write_point_cloud(src_file, tgt_sub)
        src_file = os.path.join(src_res, f'alligned_pc_lvl={node.level}_x={x}_y={y}_transformed.ply')
        o3d.io.write_point_cloud(src_file, src_sub_w_neigh)

    # choose method
    method = None
    if args.method == 'pointtopoint':
        method = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    elif args.method == 'pointtoplane':
        method = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    else:
        raise ValueError(f"The given method is wrong!\n\tGiven: {args.method}\n\tAccepted: [pointtopoint, pointtoplane]")

    max_correspondence = [0.5, 5, 4, 3, 1.5, 0.4, 0.3, 0.27, 0.25, 0.22, 0.2]
    # max_correspondence = [0.5, 5, 4, 3, 1.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    time_icp0 = time()
    reg = o3d.pipelines.registration.registration_icp(
        src_sub,
        tgt_sub,
        max_correspondence_distance=max_correspondence[node.level],
        init=np.eye(4),
        estimation_method=method
    )
    time_icp.append(time() - time_icp0)

    bbox_dict = {
        "min_bound": node.bbox.get_min_bound().tolist(),
        "max_bound": node.bbox.get_max_bound().tolist()
    }
    # print(bbox_dict)

    new_transform = np.linalg.matmul(transform, reg.transformation)
    results.append((
        node.level,
        (
            bbox_dict,
            reg.fitness,
            reg.inlier_rmse,
            new_transform
        ),
        node.is_leaf
    ))

    for child in node.children:
        run_icp_on_tree(child, pc_source, pc_target, src_res, args, new_transform, results, time_subclouds_creation, time_icp, time_transform)


if __name__ == "__main__":
    conf = OmegaConf.load("config.yaml")
    if conf.data.src_res == "default":
        conf.data.src_res = os.path.join(os.path.dirname(conf.data.src_pc1), 'results')
    
    # prepare results
    os.makedirs(conf.data.src_res, exist_ok=True)
    pointcloud_res = os.path.join(conf.data.src_res, 'pointclouds')
    if conf.args.do_output_transformed:
        os.makedirs(pointcloud_res, exist_ok=True)

    start = time()
    src_result_transforms = os.path.join(conf.data.src_res, f'pyramid_transforms_{conf.data.res_suffixe}.pickle')
    src_result_offset = os.path.join(conf.data.src_res, f'offset.txt')

    tiles = {
        'source': read_pc(conf.data.src_pc1),
        'target': read_pc(conf.data.src_pc2),
    }

    # Center pointclouds
    z_mean = tiles['source'].compute_mean_and_covariance()[0][2]
    offset = [conf.args.huge_translation[0], conf.args.huge_translation[1], z_mean]
    for tile in tiles.values():
        tile.translate(np.array([-x for x in offset]))

    # Compute normals
    if conf.args.method == 'pointtoplane':
        tiles['target'].estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=conf.args.pointtoplane_radius, 
                max_nn=conf.args.pointtoplane_max_nn,
                ))
        
    # numpy arrays
    xyz_src = np.asarray(tiles['source'].points)
    xyz_tgt = np.asarray(tiles['target'].points)

    # root bbox
    bbox = tiles['source'].get_axis_aligned_bounding_box()

    # temp
    time_quadtree_creation = 0

    # build tree
    time0 = time()
    root = build_quadtree(
        xyz_src=xyz_src,
        xyz_tgt=xyz_tgt,
        parent=None,
        bbox=bbox,
        indices_src=np.arange(len(xyz_src)),
        indices_tgt=np.arange(len(xyz_tgt)),
        indices_with_neigh=np.arange(len(xyz_src)),
        indices_sub_pts=np.arange(len(xyz_src)),
        level=0,
        max_level=conf.args.max_level,
        min_points=conf.args.min_points,
    )
    time_quadtree_creation = time() - time0

    # run ICP
    results = []
    time_subclouds_creation = []
    time_icp = []
    time_transform = []

    
    run_icp_on_tree(root, tiles['source'], tiles['target'], pointcloud_res, conf.args, np.eye(4), results, time_subclouds_creation, time_icp, time_transform)

    with open(src_result_transforms, 'wb') as f:
        pickle.dump(results, f)

    with open(src_result_offset, 'w') as f:
            f.write(f"{offset[0]},{offset[1]},{offset[2]}")

    print("Executed in ", round(time() - start, 2), " seconds")

    print("\n\t time_quadtree_creation:", time_quadtree_creation)
    print("\t time_subclouds_creation:", np.sum(time_subclouds_creation))
    print("\t time_icp:", np.sum(time_icp))
    print("\t time_transform:", np.sum(time_transform))