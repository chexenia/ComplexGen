import numpy as np
import math
import numba
import MinkowskiEngine as ME
import torch
from scipy import linalg
from typing import Optional, Union, List, Sequence, Tuple

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


#@numba.jit()        
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


def list_to_padded(
    x: Union[List[torch.Tensor], Tuple[torch.Tensor]],
    pad_size: Union[Sequence[int], None] = None,
    pad_value: float = 0.0,
    equisized: bool = False,
) -> torch.Tensor:
    r"""
    Transforms a list of N tensors each of shape (Si_0, Si_1, ... Si_D)
    into:
    - a single tensor of shape (N, pad_size(0), pad_size(1), ..., pad_size(D))
      if pad_size is provided
    - or a tensor of shape (N, max(Si_0), max(Si_1), ..., max(Si_D)) if pad_size is None.
    Args:
      x: list of Tensors
      pad_size: list(int) specifying the size of the padded tensor.
        If `None` (default), the largest size of each dimension
        is set as the `pad_size`.
      pad_value: float value to be used to fill the padded tensor
      equisized: bool indicating whether the items in x are of equal size
        (sometimes this is known and if provided saves computation)
    Returns:
      x_padded: tensor consisting of padded input tensors stored
        over the newly allocated memory.
    """
    if equisized:
        return torch.stack(x, 0)

    if not all(torch.is_tensor(y) for y in x):
        raise ValueError("All items have to be instances of a torch.Tensor.")

    # we set the common number of dimensions to the maximum
    # of the dimensionalities of the tensors in the list
    element_ndim = max(y.ndim for y in x)

    # replace empty 1D tensors with empty tensors with a correct number of dimensions
    x = [
        # pyre-fixme[16]: `Tensor` has no attribute `new_zeros`.
        (y.new_zeros([0] * element_ndim) if (y.ndim == 1 and y.nelement() == 0) else y)
        for y in x
    ]

    if any(y.ndim != x[0].ndim for y in x):
        raise ValueError("All items have to have the same number of dimensions!")

    if pad_size is None:
        pad_dims = [
            max(y.shape[dim] for y in x if len(y) > 0) for dim in range(x[0].ndim)
        ]
    else:
        if any(len(pad_size) != y.ndim for y in x):
            raise ValueError("Pad size must contain target size for all dimensions.")
        pad_dims = pad_size

    N = len(x)
    x_padded = x[0].new_full((N, *pad_dims), pad_value)
    for i, y in enumerate(x):
        if len(y) > 0:
            slices = (i, *(slice(0, y.shape[dim]) for dim in range(y.ndim)))
            x_padded[slices] = y
    return x_padded
