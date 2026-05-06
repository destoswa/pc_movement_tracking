import numpy as np


def compute_spatial_coherence(nodes, dx_list, dy_list, search_radius):
    """
    For each node, compare its displacement direction to the mean direction of neighbors.
    Returns array of dot products in [-1, 1]. Close to -1 = incoherent.
    
    Parameters:
        nodes: list of QuadNode
        dx_list, dy_list: unit vector components per node
        search_radius: spatial radius to look for neighbors (in same units as coords)
    """
    centers = np.array([
        [(n.bbox['min_bound'][0] + n.bbox['max_bound'][0]) / 2,
         (n.bbox['min_bound'][1] + n.bbox['max_bound'][1]) / 2]
        for n in nodes
    ])
    dx = np.array(dx_list)
    dy = np.array(dy_list)
    coherence = np.ones(len(nodes))

    for i in range(len(nodes)):
        dists = np.linalg.norm(centers - centers[i], axis=1)
        neighbor_mask = (dists < search_radius) & (dists > 0)
        
        if np.sum(neighbor_mask) == 0:
            coherence[i] = 1.0  # no neighbors, can't judge
            continue
        
        mean_dx = np.mean(dx[neighbor_mask])
        mean_dy = np.mean(dy[neighbor_mask])
        norm = np.sqrt(mean_dx**2 + mean_dy**2) + 1e-10
        mean_dx /= norm
        mean_dy /= norm
        
        # dot product between this node's direction and mean neighbor direction
        coherence[i] = dx[i] * mean_dx + dy[i] * mean_dy

    return coherence  # mask_incoherent = coherence < cos(45°) = 0.707


def compute_magnitude_zscore(nodes, magnitudes, search_radius):
    """
    For each node, compute z-score of its magnitude relative to spatial neighbors.
    Returns array of z-scores. High z-score = likely outlier.
    
    Parameters:
        nodes: list of QuadNode
        magnitudes: displacement magnitude per node
        search_radius: spatial radius to look for neighbors
    """
    centers = np.array([
        [(n.bbox['min_bound'][0] + n.bbox['max_bound'][0]) / 2,
         (n.bbox['min_bound'][1] + n.bbox['max_bound'][1]) / 2]
        for n in nodes
    ])
    magnitudes = np.array(magnitudes)
    zscores = np.zeros(len(nodes))

    for i in range(len(nodes)):
        dists = np.linalg.norm(centers - centers[i], axis=1)
        neighbor_mask = (dists < search_radius) & (dists > 0)
        
        if np.sum(neighbor_mask) == 0:
            zscores[i] = 0.0
            continue
        
        local_mean = np.mean(magnitudes[neighbor_mask])
        local_std = np.std(magnitudes[neighbor_mask])
        zscores[i] = (magnitudes[i] - local_mean) / (local_std + 1e-6)

    return zscores  # mask_outlier = zscores > 2.5


def compute_rotation_angles(transforms):
    """
    For each 4x4 transform, extract the rotation angle in degrees.
    Large rotation = likely artifact for terrain data.
    
    Parameters:
        transforms: list of 4x4 numpy arrays
    """
    angles = []
    for T in transforms:
        R = T[:3, :3]
        # Clamp to valid range for arccos
        cos_angle = np.clip((np.trace(R) - 1) / 2, -1, 1)
        angle = np.degrees(np.arccos(cos_angle))
        angles.append(angle)

    return np.array(angles)  # mask_large_rotation = angles > 5


def compute_confidence(coherence, zscores, angles, planarity, fitness, rmse,
                        w_coherence=1.0, w_zscore=1.0, w_rotation=1.0, 
                        w_planarity=1.0, w_fitness=1.0, w_rmse=1.0):
    """
    Compute a confidence index in [0, 1] for each node.
    1 = very confident, 0 = likely artifact.

    Parameters:
        coherence:  [-1, 1]   → 1 = coherent with neighbors
        zscores:    [0, +inf] → 0 = not an outlier
        angles:     [0, +inf] → 0 = no rotation (degrees)
        planarity:  [0, 1]    → 0 = complex geometry, 1 = flat
        fitness:    [0, 1]    → from ICP (careful: high != good, see context)
        rmse:       [0, +inf] → from ICP
    """

    # Normalize each metric to [0, 1] where 1 = confident
    c_coherence  = np.clip((coherence + 1) / 2, 0, 1)          # [-1,1] → [0,1]
    c_zscore     = np.clip(1 - zscores / 5.0, 0, 1)            # zscore=0 → 1, zscore=5 → 0
    c_rotation   = np.clip(1 - angles / 10.0, 0, 1)            # 0° → 1, 10° → 0
    c_planarity  = np.clip(1 - planarity, 0, 1)                 # flat=0 → 0, complex=1 → 1
    # c_fitness    = np.clip(1 - fitness, 0, 1)                   # inverted! high fitness = bad on flat
    # c_rmse       = np.clip(1 - rmse / np.max(rmse), 0, 1)      # low rmse = bad on flat, so also invert?

    # Weighted average
    total_weight = w_coherence + w_zscore + w_rotation + w_planarity + w_fitness + w_rmse
    confidence = (
        w_coherence * c_coherence +
        w_zscore    * c_zscore +
        w_rotation  * c_rotation +
        w_planarity * c_planarity #+
        # w_fitness   * c_fitness +
        # w_rmse      * c_rmse
    ) / total_weight

    return confidence