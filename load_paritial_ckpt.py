from mmcv.runner import _load_checkpoint,load_state_dict
import torch
from collections import OrderedDict

def load_partial_ckpt(
    model=None,
    filename=None,
    map_location='cpu',   
    logger=None,
    prefix_list=None,
    strict=False
):
    checkpoint = _load_checkpoint(filename, map_location, logger)
    # OrderedDict is a subclass of dict
    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            f'No state_dict found in checkpoint file {filename}')
    # get state_dict from checkpoint
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    new_state_dict=OrderedDict()
    for prefix in prefix_list:
        for key in state_dict:
            if key.startswith(prefix):
                new_state_dict[key] = state_dict[key]
    
    metadata = getattr(state_dict, '_metadata', OrderedDict())
    # Keep metadata in state_dict
    new_state_dict._metadata = metadata

    # load state_dict
    load_state_dict(model,new_state_dict, strict, logger)
    return checkpoint
