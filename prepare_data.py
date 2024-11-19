
import glob
import os
import point_cloud_utils as pcu
import numpy as np
from tqdm import tqdm
import fvdb
import torch
import argparse

args = argparse.ArgumentParser()
args.add_argument('--data_root', type=str, default='../data/shapenet_manifold')
args.add_argument('--num_vox', type=int, default=512)
args = args.parse_args()

data_root = args.data_root
target_root = data_root

num_vox = args.num_vox

if num_vox > 512:
    max_num_vox = num_vox
    sample_pcs_num = 5_000_000
else:
    max_num_vox = 512
    sample_pcs_num = 1_000_000
vox_size = 1.0 / max_num_vox

target_dir = target_root
os.makedirs(target_dir, exist_ok=True)

model_list = glob.glob(os.path.join(data_root, "*.obj"))

for model_path in tqdm(model_list):
    model_name = os.path.basename(model_path).split(".")[0]
    target_path = os.path.join(target_dir, f"{model_name}.pkl")
    # check if target_path exist
    if os.path.exists(target_path):
        continue
    
    v, f = pcu.load_mesh_vf(os.path.join(model_path))
    
    try:
        fid, bc = pcu.sample_mesh_random(v, f, sample_pcs_num)
        ref_xyz = pcu.interpolate_barycentric_coords(f, fid, bc, v)
    except:
        fid, bc = pcu.sample_mesh_random(v, f, sample_pcs_num)
        ref_xyz = pcu.interpolate_barycentric_coords(f, fid, bc, v)

    n = pcu.estimate_mesh_face_normals(v, f)
    ref_normal = n[fid]
            
    ijk = pcu.voxelize_triangle_mesh(v, f.astype(np.int32), vox_size, np.zeros(3))
    grid = fvdb.sparse_grid_from_ijk(fvdb.JaggedTensor([torch.from_numpy(ijk).cuda()]), voxel_sizes=vox_size, origins=[vox_size / 2.] * 3)
    
    # get normal ref
    ref_xyz = torch.from_numpy(ref_xyz).float().cuda()
    ref_normal = torch.from_numpy(ref_normal).float().cuda()
    input_normal = grid.splat_trilinear(fvdb.JaggedTensor(ref_xyz), fvdb.JaggedTensor(ref_normal))
    # normalize normal
    input_normal.jdata /= (input_normal.jdata.norm(dim=1, keepdim=True) + 1e-6) # avoid nan

    # _, f_idx, _ = pcu.closest_points_on_mesh(
    #     grid.grid_to_world(grid.ijk.float()).jdata.cpu().numpy().astype(float), v.astype(float), f)
    # input_normal = fvdb.JaggedTensor([torch.from_numpy(n[f_idx])])        
            
    # normalize xyz to conv-onet scale
    xyz = grid.grid_to_world(grid.ijk.float()).jdata
    xyz_norm = xyz * 128 / 100
    ref_xyz = ref_xyz * 128 / 100
    
    # convert to fvdb_grid format
    if num_vox == 512:
        # not splatting
        target_voxel_size = 0.0025
        target_grid = fvdb.sparse_grid_from_points(
                fvdb.JaggedTensor(xyz_norm), voxel_sizes=target_voxel_size, origins=[target_voxel_size / 2.] * 3)
    elif num_vox == 16:
        # splatting
        target_voxel_size = 0.08
        target_grid = fvdb.sparse_grid_from_nearest_voxels_to_points(
                    fvdb.JaggedTensor(xyz_norm), voxel_sizes=target_voxel_size, origins=[target_voxel_size / 2.] * 3)
    elif num_vox == 128:
        # splatting
        target_voxel_size = 0.01
        target_grid = fvdb.sparse_grid_from_nearest_voxels_to_points(
                    fvdb.JaggedTensor(xyz_norm), voxel_sizes=target_voxel_size, origins=[target_voxel_size / 2.] * 3)
    elif num_vox == 256:
        target_voxel_size = 0.005
        target_grid = fvdb.sparse_grid_from_nearest_voxels_to_points(
                    fvdb.JaggedTensor(xyz_norm), voxel_sizes=target_voxel_size, origins=[target_voxel_size / 2.] * 3)
    elif num_vox == 1024:
        target_voxel_size = 0.00125
        target_grid = fvdb.sparse_grid_from_points(
                    fvdb.JaggedTensor(xyz_norm), voxel_sizes=target_voxel_size, origins=[target_voxel_size / 2.] * 3)
    else:
        raise NotImplementedError
    
    # get target normal
    target_normal = target_grid.splat_trilinear(fvdb.JaggedTensor(ref_xyz), fvdb.JaggedTensor(ref_normal))
    target_normal.jdata /= (target_normal.jdata.norm(dim=1, keepdim=True) + 1e-6)

    save_dict = {
        "points": target_grid.to("cpu"),
        "normals": target_normal.cpu(),
        "ref_xyz": ref_xyz.cpu(),
        "ref_normal": ref_normal.cpu(),
    }
    
    torch.save(save_dict, target_path)
