import os
import sys

from tqdm import tqdm
path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, path)

import os
import argparse
import importlib
import pickle

import torch
import torch.backends.cudnn as cudnn

from datetime import datetime
from mmcv import Config, DictAction
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmdet.apis import set_random_seed
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model

from loaders.builder import build_dataloader
from models.utils import sparse2dense

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize results')
    parser.add_argument('--config', required=True, help='Path to config file')
    parser.add_argument('--weights', required=True, help='Path to checkpoint')
    parser.add_argument('--save-dir', type=str, default='visualizations', help='Visualize results')
    args = parser.parse_args()
    
    save_dir = os.path.join(args.save_dir, 'vis_2','occ_pred')
    os.makedirs(save_dir,exist_ok=True)

    # parse configs
    cfgs = Config.fromfile(args.config)
    
    # register custom module
    importlib.import_module('models')
    importlib.import_module('loaders')

    set_random_seed(0, deterministic=True)
    cudnn.benchmark = True

    for p in cfgs.data.val.pipeline:
        if p['type'] == 'LoadMultiViewImageFromMultiSweeps':
            p['force_offline'] = True
    val_dataset = build_dataset(cfgs.data.val)
    val_loader = build_dataloader(
        val_dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfgs.data.workers_per_gpu,
        num_gpus=1,
        dist=False,
        shuffle=False,
        seed=0,
    )

    model = build_model(cfgs.model)
    model.cuda()
    model = MMDataParallel(model, [0])

    load_checkpoint(model, args.weights, map_location='cuda', strict=True)
    model.eval()

    
    with torch.no_grad():
        for i, data in tqdm(enumerate(val_loader)):
            
            result_dict = model(return_loss=False, rescale=True, **data)
            
            voxel_semantics = data['voxel_semantics'][0]
            mask_camera = data['mask_camera'][0]
            
            for b in range(len(result_dict)):
                
                occ_loc = result_dict[b]['occ_loc']
                sem_pred = result_dict[b]['sem_pred']

                occ_gt = voxel_semantics[b].cpu().numpy() # (200 200 16)
                mask_cam = mask_camera[b].cpu().numpy()   # (200 200 16)

                occ_pred, _ = sparse2dense(
                    occ_loc,       # (56991 3)
                    sem_pred,      # (56991,)
                    dense_shape=occ_gt.shape, # (200 200 16)
                    empty_value=17)               # occ_pred: (200 200 16)                
                
                output = dict(
                    occ_pred = occ_pred,
                    occ_gt = occ_gt,
                    occ_mask = mask_cam,
                )                    
                save_name = f"{i}_{b}.pkl"
                save_path = os.path.join(save_dir, save_name)
                with open(save_path, 'wb') as f:
                    pickle.dump(output, f)  