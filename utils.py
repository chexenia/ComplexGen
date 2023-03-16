import numpy as np
import math
import numba
import MinkowskiEngine as ME
import torch
from scipy import linalg

th_norm = 1e-6

def normalize_model(points_with_normal, to_unit_sphere=True):
    assert(len(points_with_normal.shape) == 2 and points_with_normal.shape[1] == 6)
    points = points_with_normal[:,:3]
    normal = points_with_normal[:,3:]
    #normalize to unit bounding box
    max_coord = points.max(axis=0)
    min_coord = points.min(axis=0)
    center = (max_coord + min_coord) / 2.0
    scale = (max_coord - min_coord).max()
    normalized_points = points - center
    if(to_unit_sphere):
      scale = math.sqrt(np.square(normalized_points).sum(-1).max())*2
    # normalized_points *= 0.95/scale
    normalized_points *= 1.0/scale
    return np.concatenate([normalized_points, normal], axis=1), -center, 1.0/scale


@numba.jit()        
def points2sparse_voxel(points_with_normal, voxel_dim, feature_type, with_normal, pad1s):
    #covert to COO format, assume input points is already normalize to [-0.5 0.5]
    points = points_with_normal[:,:3] + 0.5
    voxel_dict = {}
    voxel_length = 1.0 / voxel_dim
    voxel_coord = np.clip(np.floor(points / voxel_length).astype(np.int32), 0, voxel_dim-1)
    points_normal_norm = linalg.norm(points_with_normal[:,3:], axis=1, keepdims=True)
    points_normal_norm[points_normal_norm < th_norm] = th_norm
    if(feature_type == 'local'):
      local_coord = (points - voxel_coord.astype(np.float32)*voxel_length)*voxel_dim - 0.5
      local_coord = np.concatenate([local_coord, points_with_normal[:,3:] / points_normal_norm, np.ones([local_coord.shape[0], 1])], axis=-1)
    elif(feature_type == 'global'):
      local_coord = points - 0.5
      local_coord = np.concatenate([local_coord, points_with_normal[:,3:] / points_normal_norm, np.ones([local_coord.shape[0], 1])], axis=-1)
    
    stat_voxel_dict = {}
    
    for i in range(voxel_coord.shape[0]):
      coord_tuple = (voxel_coord[i,0], voxel_coord[i,1], voxel_coord[i,2])
      if(coord_tuple not in voxel_dict):
        voxel_dict[coord_tuple] = local_coord[i]
      else:
        voxel_dict[coord_tuple] += local_coord[i]
    
    locations = np.array(list(voxel_dict.keys()))
    features = np.array(list(voxel_dict.values()))
    points_in_voxel = features[:,6:]
    features = features / points_in_voxel #pad ones
    position = features[:,:3]
    normals = features[:,3:6]
    pad_ones = features[:,6:]
    normals /= linalg.norm(normals, axis=-1, keepdims=True) + 1e-10
    
    '''
    max_variance = 0
    max_variance_signals = None
    #do the statistics
    for item in stat_voxel_dict:
      if(len(stat_voxel_dict[item]) == 1): continue
      mean_normal = voxel_dict[item][3:6]
      mean_normal /= linalg.norm(mean_normal)
      voxel_normals = np.stack(stat_voxel_dict[item], axis=0)[:,3:6]
      voxel_normals /= linalg.norm(voxel_normals, axis=1, keepdims=True)
      diff = np.square(voxel_normals - np.reshape(mean_normal, [-1, 3])).sum(-1).mean()
      if(diff > max_variance):
        max_variance = diff
        max_variance_signals = stat_voxel_dict[item]
    
    print("max variance voxel {} {}".format(max_variance, max_variance_signals))
    '''
    if(with_normal and pad1s):
      features = np.concatenate([position, normals, pad_ones], axis=1)
    elif(pad1s):
      features = np.concatenate([position, pad_ones], axis=1)
    elif(with_normal):
      features = np.concatenate([position, normals], axis=1)
    else:
      features = position
    
    return locations.astype(np.int32), features.astype(np.float32)


def points2sparse_voxel_mink(points_with_normal, voxel_dim, feature_type, with_normal, pad1s):
    """
      Use Minkowski Engine's native sparse tensorfield algorithm to voxelize the point cloud.
    """
    voxel_size = 1.0 / voxel_dim
    coords = points_with_normal[:,:3] + 0.5
    assert (feature_type == 'global')
    fea = points_with_normal[:,:3]
    if with_normal:
      points_normal_norm = linalg.norm(points_with_normal[:,3:], axis=1, keepdims=True)
      points_normal_norm[points_normal_norm < th_norm] = th_norm
      fea = np.concatenate([fea, points_with_normal[:,3:] / points_normal_norm], axis=-1)
    if pad1s:
      fea = np.concatenate([fea, np.ones([fea.shape[0], 1])], axis=-1)

    sinput = ME.SparseTensor(
      features=torch.from_numpy(fea), # Convert to a tensor
      coordinates=ME.utils.batched_coordinates([coords / voxel_size]),  # coordinates must be defined in a integer grid. If the scale
      quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE  # when used with continuous coordinates, average features in the same coordinate
    ).detach()
    return sinput.coordinates_at(0).numpy().astype(np.int32), sinput.features_at(0).numpy().astype(np.float32)


def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
    """ Randomly jitter points. jittering is per point.
      Input:
        BxNx3 array, original batch of point clouds
      Return:
        BxNx3 array, jittered batch of point clouds
    """
    B, N, C = batch_data.shape
    assert (clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1 * clip, clip)
    jittered_data += batch_data
    return jittered_data.astype(np.float32)
