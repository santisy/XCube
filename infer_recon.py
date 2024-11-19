import importlib
import argparse
import glob
import os
import yaml

import torch
from easydict import EasyDict as edict
from omegaconf import OmegaConf 
from pathlib import Path
from tqdm import tqdm

import nksr
import trimesh
from xcube.models.autoencoder import Model as VAE
from xcube.data.base import DatasetSpec as DS
from xcube.utils import exp

def create_model_from_args(config_path, ckpt_path, strict=True):
    model_yaml_path = Path(config_path)
    model_args = exp.parse_config_yaml(model_yaml_path)
    net_module = importlib.import_module("xcube.models." + model_args.model).Model
    args_ckpt = Path(ckpt_path)
    assert args_ckpt.exists(), "Selected checkpoint does not exist!"
    net_model = net_module.load_from_checkpoint(args_ckpt, hparams=model_args, strict=strict)
    return net_model.eval()

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", required=True, type=str)
parser.add_argument("--input_dir", required=True, type=str)
parser.add_argument("--class_label", required=True, type=str)

args = parser.parse_args()

# Prepare directories
input_dir = args.input_dir
output_dir = os.path.join("recon_out", os.path.basename(input_dir))
os.makedirs(output_dir, exist_ok=True)

if args.class_label == "chair":
    config_nksr = "configs/shapenet/chair/train_nksr_refine.yaml"
    ckpt_nksr = "checkpoints/chair/nksr_refine/last.ckpt"
elif args.class_label == "plane":
    config_nksr = "configs/shapenet/plane/train_nksr_refine.yaml"
    ckpt_nksr = "checkpoints/plane/nksr_refine/last.ckpt"
elif args.class_label == "objaverse":
    config_nksr = "configs/objaverse/train_nksr_refine.yaml"
    ckpt_nksr = "checkpoints/objaverse/nksr_refine/last.ckpt"

# Construct the reconstructor
reconstructor = create_model_from_args(config_nksr, ckpt_nksr, strict=False).cuda()

input_files = glob.glob(os.path.join(input_dir, "*.pkl"))

hparams = OmegaConf.load("configs/train/vae/vae_128x128x128_sparse.yaml")
if args.class_label == "objaverse":
    hparams_v2 = OmegaConf.load("configs/objaverse/train_vae_128x128x128_sparse.yaml")
    hparams = OmegaConf.merge(hparams, hparams_v2)

hparams.pretrained_weight = args.ckpt
vae = VAE(hparams).cuda()
vae = vae.eval()
device = torch.device("cuda")

# Data preparation
for data_path in tqdm(input_files):
    data_name = os.path.basename(data_path).split(".")[0]
    input_data = torch.load(data_path)

    data_dict = {}
    data_dict[DS.TARGET_NORMAL] = input_data['normals'].to(device)
    data_dict[DS.INPUT_PC] = input_data['points'].to(device)

    out_dict = {}
    with torch.no_grad():
        vae(data_dict, out_dict)

    output_x = out_dict['x']
    res = out_dict

    # Export the mesh
    batch_idx = 0
    pd_grid = output_x.grid[batch_idx]
    pd_normal = res["normal_features"][-1].feature[batch_idx].jdata
    with torch.no_grad():
        field = reconstructor.forward({'in_grid': pd_grid, 'in_normal': pd_normal})
    field = field['neural_udf']
    mesh = field.extract_dual_mesh(max_depth=0, grid_upsample=2)
    # save mesh
    mesh_save = trimesh.Trimesh(vertices=mesh.v.detach().cpu().numpy(),
                                faces=mesh.f.detach().cpu().numpy())
    # mesh_save.merge_vertices()
    mesh_save.export(os.path.join(output_dir, f"{data_name}_XCubeRecon.obj"))

