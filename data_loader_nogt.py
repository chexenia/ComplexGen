import numpy as np
import os
import torch
import random
import open3d as o3d
from pathlib import Path
from torch.utils.data.distributed import DistributedSampler
from scipy.spatial.transform import Rotation as R
from utils import *
average_patch_area = 0
average_squared_curve_length = 0
pack_size = 10000
th_norm = 1e-6
points_per_curve_dim = 34
flag_normal_noise = True
r_normal_noise = 0.2

class HDDataset(torch.utils.data.Dataset):
    def __init__(self, data_root, voxel_dim, feature_type='local', pad1s=True, random_rotation=False, random_angle = False, with_normal=True, flag_quick_test = False, flag_noise = 0, flag_grid = False, num_angles = 4, flag_patch_uv = False, flag_backbone = False, dim_grid = 10, eval_res_cov = False):
        self.data_root = Path(data_root)
        self.voxel_dim = voxel_dim
        self.feature_type = feature_type
        assert(self.feature_type == 'global' or self.feature_type == 'local' or self.feature_type == 'occupancy')
        self.pad1s = pad1s
        self.random_rotation_augmentation = random_rotation
        self.random_angle = random_angle
        if(self.random_rotation_augmentation): print("enable rotation augmentation")
        self.with_normal = with_normal
        self.flag_quick_test = flag_quick_test
        self.flag_noise = flag_noise
        self.flag_grid = flag_grid
        self.num_angles = num_angles
        self.flag_patch_uv = flag_patch_uv
        self.flag_backbone = flag_backbone
        self.dim_grid = dim_grid
        self.eval_res_cov = eval_res_cov
        self.fourteen_mat = []
        for i in range(4):
          self.fourteen_mat.append(R.from_rotvec(np.pi/2 * i * np.array([0,1,0])).as_matrix())
        self.fourteen_mat.append(R.from_rotvec(np.pi/2 * 1 * np.array([1,0,0])).as_matrix())
        self.fourteen_mat.append(R.from_rotvec(np.pi/2 * 3 * np.array([1,0,0])).as_matrix())
        c = np.sqrt(3)/3
        s = -np.sqrt(6)/3
        cornerrot1 = np.array([[c,0,-s],[0,1,0],[s,0,c]])
        for i in range(4):
          self.fourteen_mat.append( np.matmul(R.from_rotvec((np.pi/2 * i + np.pi / 4) * np.array([0,0,1])).as_matrix(), cornerrot1).transpose() )
        
        c = -np.sqrt(3)/3
        cornerrot2 = np.array([[c,0,-s],[0,1,0],[s,0,c]])
        for i in range(4):
          self.fourteen_mat.append( np.matmul(R.from_rotvec((np.pi/2 * i + np.pi / 4) * np.array([0,0,1])).as_matrix(), cornerrot2).transpose() )
        if(self.with_normal): print("normal is included in insput signal")
           
        from glob import glob
        filepaths = glob(str(self.data_root)+'/**/*.ply', recursive=True)
        self.filepaths = [Path(filepath) for filepath in filepaths]
        print(self.filepaths)

    def __len__(self):
        return len(self.filepaths)
    
    #only worked when no rotation augmentation is used
    # @lru_cache(maxsize=None)
    def __getitem__(self, idx):
        filepath = self.filepaths[idx % len(self)]
        filename = filepath.stem
        mesh = o3d.io.read_triangle_mesh(str(filepath))
        if not mesh.has_vertex_normals():
            mesh.compute_vertex_normals()
        points = np.asarray(mesh.vertices, dtype=np.float32)
        normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
        points_normals = np.concatenate((points, normals), axis=1)
        item_points, _, _ = normalize_model(points_normals)

        if self.flag_backbone:
          item_points_ori = np.copy(item_points)

        if self.flag_noise > 0:
          if self.flag_noise == 1:
            sigma=0.01
          elif self.flag_noise == 2:
            sigma =0.02
            # print('noise level 2')
          elif self.flag_noise == 3:
            sigma = 0.05 

          clip= 5.0 * sigma
          jittered_data_pts = np.clip(sigma * np.random.randn(item_points.shape[0],3), -1 * clip, clip)
          item_points[:,:3] = item_points[:,:3] + jittered_data_pts

          if flag_normal_noise:
            normal_noise = np.random.random_sample((item_points.shape[0],3)) *2 -1
            normal_noise_norm = np.linalg.norm(normal_noise, axis =-1).reshape(-1,1)
            normal_noise_norm[normal_noise_norm < th_norm] = th_norm
            normal_noise = normal_noise / normal_noise_norm
            new_normal = item_points[:,3:] + normal_noise * r_normal_noise
            new_normal_norm = np.linalg.norm(new_normal, axis = -1).reshape(-1,1)
            new_normal_norm[new_normal_norm < th_norm] = th_norm
            item_points[:, 3:] = new_normal / new_normal_norm

        if(self.random_rotation_augmentation):
          if not self.random_angle:
            if self.num_angles == 4:
              rot_z = R.from_rotvec(np.pi/2 * random.randint(0,3) * np.array([0,0,1])).as_matrix()
              rot = rot_z
            elif self.num_angles == 56:
              rot = self.fourteen_mat[random.randint(0,13)]
              rot_z = R.from_rotvec(np.pi/2 * random.randint(0,3) * np.array([0,0,1])).as_matrix()
              rot = np.matmul(rot_z, rot)
            elif self.num_angles == 14:
              rot = self.fourteen_mat[random.randint(0,13)]
            elif self.num_angles == -1:
              rotation_angle = np.random.uniform() * 2 * np.pi
            item_points_ori = np.dot(item_points_ori, rot) #apply the transform, save as matmul
            item_points_ori = np.reshape(item_points_ori, [-1,6])
                        
        locations, features = points2sparse_voxel(item_points, self.voxel_dim, self.feature_type, self.with_normal, self.pad1s)
        if self.flag_backbone:
          locations_ori, features_ori = points2sparse_voxel(item_points_ori, self.voxel_dim, self.feature_type, self.with_normal, self.pad1s)
          return (locations, features, locations_ori, features_ori, filename, idx)

        return (locations, features, item_points, filename, idx)


def test_data_loader_nogt(batch_size=32, voxel_dim=128, feature_type='local', pad1s=True, data_folder=None, rotation_augmentation=False, random_angle = False, with_normal=True, with_distribute_sampler=False, flag_quick_test = False, flag_noise = 0, flag_grid = False, num_angle = 4, flag_patch_uv = False, flag_backbone = False, dim_grid = 10, eval_res_cov = False):
    #default parameters:
    #input_feature_type: global
    #backbone_feature_encode: false
    #rotation_augment: false
    #input normal signal: false
    #with distribute sampler: true

  def collate_function(tensorlist):
    batch_size = len(tensorlist)
    locations = [np.concatenate([tensorlist[i][0], np.ones([tensorlist[i][0].shape[0], 1], dtype=np.int32)*i], axis=-1) for i in range(batch_size)]
    features = [tensorlist[i][1] for i in range(batch_size)]
    if flag_backbone:
      locations_ori = [np.concatenate([tensorlist[i][2], np.ones([tensorlist[i][2].shape[0], 1], dtype=np.int32)*i], axis=-1) for i in range(batch_size)]
      features_ori = [tensorlist[i][3] for i in range(batch_size)]
      input_sample_idx = [tensorlist[i][5] for i in range(batch_size)]
      return torch.from_numpy(np.concatenate(locations, axis=0)), torch.from_numpy(np.concatenate(features, axis=0)),\
      torch.from_numpy(np.concatenate(locations_ori, axis=0)), torch.from_numpy(np.concatenate(features_ori, axis=0)), input_sample_idx

    input_sample_idx = [tensorlist[i][4] for i in range(batch_size)]
    input_pointcloud = [tensorlist[i][2] for i in range(batch_size)]
    return torch.from_numpy(np.concatenate(locations, axis=0)), torch.from_numpy(np.concatenate(features, axis=0)),\
           input_pointcloud, input_sample_idx
  
  dataset = HDDataset(data_folder, voxel_dim, feature_type=feature_type, pad1s=pad1s, random_rotation=rotation_augmentation, random_angle = random_angle, with_normal=with_normal, flag_quick_test=flag_quick_test, flag_noise=flag_noise, flag_grid = flag_grid, num_angles = num_angle, flag_patch_uv = flag_patch_uv, flag_backbone = flag_backbone, dim_grid = dim_grid, eval_res_cov=eval_res_cov)
  if(with_distribute_sampler): #train mode true
    sampler = DistributedSampler(dataset)
    data = torch.utils.data.DataLoader(dataset, batch_size=batch_size, collate_fn=collate_function, drop_last=True, sampler=sampler)
    return data, sampler
  else:
    data = torch.utils.data.DataLoader(dataset, batch_size=batch_size, collate_fn=collate_function, drop_last=True, shuffle=False, num_workers=4)
    return data
    