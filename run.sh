
pip install torch-scatter -f https://data.pyg.org/whl/torch-{Version}+${CUDA}.html # Change the torch and cuda version
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.1+cu121.html
pip install fvcore einops
pip install spconv-cu121

mim run mmdet3d create_data nuscenes --root-path /c20250502/wangyushen/Datasets/NuScenes/v1.0-trainval --out-dir /c20250502/wangyushen/Datasets/NuScenes/v1.0-trainval --extra-tag nuscenes

python /vepfs-mlp2/c20250502/haoce/wangyushen/OPUS/scripts/gen_sweep_info.py \
    --data-root /c20250502/wangyushen/Datasets/NuScenes/v1.0-trainval


# opusv1
python /vepfs-mlp2/c20250502/haoce/wangyushen/OPUS/val.py \
    --config /vepfs-mlp2/c20250502/haoce/wangyushen/OPUS/configs/customs/opusv1-s_r50_704x256_8f_nusc-occ3d_100e.py \
    --weights /c20250502/wangyushen/Weights/opus/opusv1-s_r50_704x256_8f_nusc-occ3d_100e.pth

python /vepfs-mlp2/c20250502/haoce/wangyushen/OPUS/demo.py \
    --config /vepfs-mlp2/c20250502/haoce/wangyushen/OPUS/configs/customs/opusv1-s_r50_704x256_8f_nusc-occ3d_100e.py \
    --weights /c20250502/wangyushen/Weights/opus/opusv1-s_r50_704x256_8f_nusc-occ3d_100e.pth

# opusv1-fusion
python /vepfs-mlp2/c20250502/haoce/wangyushen/OPUS/val.py \
    --config /vepfs-mlp2/c20250502/haoce/wangyushen/OPUS/configs/customs/opusv1-fusion-s_r50_704x256_8f_nusc-occ3d_100e.py \
    --weights /c20250502/wangyushen/Weights/opus/opusv1-fusion-s_r50_704x256_8f_nusc-occ3d_100e.pth

# opusv2
python /vepfs-mlp2/c20250502/haoce/wangyushen/OPUS/val.py \
    --config /vepfs-mlp2/c20250502/haoce/wangyushen/OPUS/configs/customs/opusv2-s_r50_704x256_8f_nusc-occ3d_100e.py \
    --weights /c20250502/wangyushen/Weights/opus/opusv2-s_r50_704x256_8f_nusc-occ3d_100e.pth