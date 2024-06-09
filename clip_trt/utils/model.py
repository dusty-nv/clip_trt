#!/usr/bin/env python3
import os
import json

import torch
import tensorrt


def clip_model_type(model, types=['clip', 'siglip']):
    """
    Determine if this is a CLIP or SigLIP model, either by checking for those
    in the model name or path, or by loading its config.json and looking there.
    """
    def search_types(model_name):
        for key in types:
            if key in model_name.lower():
                return key 
        return None
        
    # for model names, or paths containing the name
    model_type = search_types(model)  
    
    if model_type:
        return model_type
        
    # for paths without, check the config.json
    if not os.path.isdir(model):  
        return None
        
    try:
        config_path = os.path.join(model, 'config.json')
        with open(config_path) as config_file:
            return search_types(json.load(config_file)['model_type'])
    except Exception as error:
        logging.error(f"failed to get CLIP model type type from local model config under {model} ({error})")

    
def trt_model_filename(model, suffix=None, ext='.pt'):
    """
    Returns a filename for a TensorRT engine that includes the TensorRT version and CUDA device SM.
    This is used so that the cached TensorRT engines only get used with the same version of TensorRT and GPU.
    """
    model = model.replace('/','-').replace('@','-')
    trt_version = tensorrt.__version__.replace('.', '')
    cuda_sm = torch.cuda.get_device_capability()
    cuda_sm = f"sm{cuda_sm[0]}{cuda_sm[1]}"
    suffix = f"_{suffix}" if suffix else ''
    return f"{model}_trt-{trt_version}_{cuda_sm}{suffix}{ext}"
    
