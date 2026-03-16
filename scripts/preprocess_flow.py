import sys
import os.path as osp
path = osp.abspath(osp.join(osp.dirname(__file__), '..'))
sys.path.insert(0, path)

import pickle
import argparse
import numpy as np
import torch
from collections import defaultdict
from nuscenes.eval.common.utils import Quaternion
from nuscenes.utils.geometry_utils import transform_matrix
# from mmcv.ops import 


def occ3d_loader(args, data):
    scene_name, sample_token = data['scene_name'], data['token']
    occ_file = osp.join(args.occ_root, scene_name, sample_token, 'labels.npz')
    occ = np.load(occ_file)['semantics']
    name_mapper = np.array([
        'others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
        'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
        'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
        'vegetation'], dtype=np.dtype('<U20'))

    occ = torch.tensor(occ).long().cuda()
    pc_range = occ.new_tensor([-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]).float()
    voxel_size = occ.new_tensor([0.4, 0.4, 0.4]).float()
    scene_size = pc_range[3:] - pc_range[:3]

    device = occ.device
    W, H, Z = occ.shape
    x = torch.arange(0, W, dtype=torch.float32, device=device)
    cx = (x + 0.5) / W * scene_size[0] + pc_range[0]
    y = torch.arange(0, H, dtype=torch.float32, device=device)
    cy = (y + 0.5) / H * scene_size[1] + pc_range[1]
    z = torch.arange(0, Z, dtype=torch.float32, device=device)
    cz = (z + 0.5) / Z * scene_size[2] + pc_range[2]

    xx = x[:, None, None].expand(W, H, Z)
    yy = y[None, :, None].expand(W, H, Z)
    zz = z[None, None, :].expand(W, W, Z)
    index = torch.stack([xx, yy, zz], dim=-1)

    cxx = cx[:, None, None].expand(W, H, Z)
    cyy = cy[None, :, None].expand(W, H, Z)
    czz = cz[None, None, :].expand(W, W, Z)
    coors = torch.stack([cxx, cyy, czz], dim=-1)

    index = index[occ != 17]
    coors = coors[occ != 17]
    names = name_mapper[occ[occ != 17].cpu().numpy()]
    return index, coors, names, np.eye(4)

def occ3d_saver():
    pass


def occupancy_loader(data_root):
    raise NotImplementedError


def occupancy_saver():
    raise NotImplementedError


def collect_boxes(other, base):
    if other is None:
        return 
    
    other_l2e_t = other['lidar2ego_translation']
    other_l2e_r = other['lidar2ego_rotation']
    other_l2e_mat = transform_matrix(other_l2e_t, Quaternion(other_l2e_r))
    other_e2g_t = other['ego2global_translation']
    other_e2g_r = other['ego2global_rotation']
    other_e2g_mat = transform_matrix(other_e2g_t, Quaternion(other_e2g_r))
    other_l2g_mat = other_e2g_mat @ other_l2e_mat

    base_l2e_t = base['lidar2ego_translation']
    base_l2e_r = base['lidar2ego_rotation']
    base_e2l_mat = transform_matrix(base_l2e_t, Quaternion(base_l2e_r), inverse=True)
    base_e2g_t = base['ego2global_translation']
    base_e2g_r = base['ego2global_rotation']
    base_g2e_mat = transform_matrix(base_e2g_t, Quaternion(base_e2g_r), inverse=True)
    base_g2l_mat = base_e2l_mat @ base_g2e_mat

    other2base = base_g2l_mat @ other_l2g_mat
    delta_yaw = np.atan2(other2base[1, 0], other2base[0, 0])

    boxes, names = other['gt_boxes'], other['gt_names']
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize results')
    parser.add_argument('--dataset', choices=['occ3d', 'occupancy'], default='occ3d')
    parser.add_argument('--data_root', default='data/nuscenes/', type=str, help='Path to config file')
    parser.add_argument('--occ_root', default='data/nuscenes/gts', type=str, help='Path to config file')
    parser.add_argument('--ann_files', nargs='+', type=str,
                        default=['nuscenes_infos_train_sweep2.pkl', 'nuscenes_infos_val_sweep2.pkl'])
    parser.add_argument('--his_n_seq', default=10, type=int, help='Path to config file')
    parser.add_argument('--fut_n_seq', default=10, type=int, help='Path to config file')
    args = parser.parse_args()

    data = []
    for ann_file in args.ann_files:
        res = pickle.load(open(osp.join(args.data_root, ann_file), 'rb'))
        data.extend(res['infos'])

    if args.dataset == 'occ3d':
        loader, saver = occ3d_loader, occ3d_saver
    elif args.dataset == 'occupancy':
        loader, saver = occupancy_loader, occupancy_saver
    else:
        raise NotImplementedError
    
    scene_collector = defaultdict(list)
    for d in data:
        scene_collector[d['scene_name']].append(d)
    for value in scene_collector.values():
        value.sort(key=lambda x: x['timestamp'])
    
    for scene_name, scene_data in scene_collector.items():
        for i in range(len(scene_data)):
            curr_frame = scene_data[i]
            his_frames = [None if i - n < 0 else scene_data[i - n]
                          for n in range(-args.his_n_seq, 0)]
            his_boxes_infos = [collect_boxes(f, curr_frame) for f in his_frames]
            fut_frames = [None if i + n > len(scene_data) - 1 else scene_data[i+n]
                          for n in range(1, args.fut_n_seq + 1)]
            fut_boxes_infos = [collect_boxes(f, curr_frame) for f in fut_frames]

            scene_data = scene_collector[d['scene_name']]
            import pdb; pdb.set_trace()
            # index, coors, names, ego2occ = loader(args, data[i])

