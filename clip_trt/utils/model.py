#!/usr/bin/env python3
import torch
import tensorrt


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
    
