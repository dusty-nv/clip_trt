#!/usr/bin/env python3
import time
import torch
import logging
import argparse

from termcolor import cprint

from clip_trt import CLIPModel
from clip_trt.utils import LogFormatter, load_prompts, print_table


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--model", type=str, default="openai/clip-vit-large-patch14-336", help="the name or path of the CLIP/SigLIP model to load")
parser.add_argument("--inputs", type=str, action='append', nargs='*', help="assets to compute the CLIP embeddings for - can be image filenames or URLs load (can specify multiple files or wildcards), and text strings or .txt/.json files")

parser.add_argument("--image-scaling", type=str, default=None, choices=['crop', 'resize'], help="override the default image rescaling method ('crop' for CLIP and 'resize' for SigLIP)")
parser.add_argument("--logit-scaling", type=str, default='off', choices=['on', 'off'], help="enable output logit scaling and bias")
parser.add_argument("--probabilities", type=str, default='off', choices=['on', 'off'], help="apply activation function to output logits (softmax for CLIP and sigmoid for SigLIP)")
parser.add_argument("--normalization", type=str, default='on',  choices=['on', 'off'], help="apply L2 normalization to the embeddings")

parser.add_argument("--disable-projector", action='store_true', help="disable the projection layer in CLIP models")
parser.add_argument("--disable-trt", dest='use_tensorrt', action='store_false', help="disable using TensorRT and fall back to Transformers API")
parser.add_argument("--trt-cache", type=str, default="~/.cache/clip_trt", help="directory to save TensorRT engines under")

parser.add_argument("--log-level", type=str, default='info', choices=['debug', 'info', 'warning', 'error', 'critical'], help="the logging level to stdout")
parser.add_argument("--debug", "--verbose", action="store_true", help="set the logging level to debug/verbose mode")

args = parser.parse_args()

if args.debug:
    args.log_level = "debug"
    
LogFormatter.config(level=args.log_level)
torch.set_printoptions(sci_mode=False, precision=4)

print(args)

# load inputs
inputs = args.inputs

for i, input in enumerate(inputs):
    inputs[i] = load_prompts(input)

# load model
model = CLIPModel.from_pretrained(
    args.model,
    projector=False if args.disable_projector else None,
    crop=args.image_scaling,
    use_tensorrt=args.use_tensorrt,
    trt_cache=args.trt_cache
)

print_table(model.vision.config)
        
# get similarity scores
logits = model(
    *inputs,
    normalize=(args.normalization == 'on'),
    logit_scaling=(args.logit_scaling == 'on'),
    return_probs=(args.probabilities == 'on'),
)

print('Inputs:', inputs)
print('Similarity:', logits.shape, logits.dtype, logits.device, '\n', logits)

