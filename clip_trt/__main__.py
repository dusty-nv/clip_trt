#!/usr/bin/env python3
import time
import logging
import argparse

from termcolor import cprint

from clip_trt import CLIPModel
from clip_trt.utils import LogFormatter, print_table

# see utils/args.py for options
parser = ArgParser()

self.add_argument("prompts", type=str, nargs='*', help="the assets to compute the CLIP embeddings for - can be image filenames or URLs load (can specify multiple files or wildcards), and text strings or .txt/.json files")

self.add_argument("--model", type=str, default="openai/clip-vit-large-patch14-336", help="the name or path of the CLIP/SigLIP model to load")
self.add_argument("--api", type=str, default='trt', choices=['hf', 'trt'], help="select to use either TensorRT or HuggingFace Transformers")

self.add_argument("--scaling", type=str, default=None, choices=['crop', 'resize'], help="override the default image rescaling method ('crop' for CLIP and 'resize' for SigLIP)")

self.add_argument("--trt-cache", type=str, default="~/.cache/clip_trt", help="directory to save TensorRT engines under")
self.add_argument("--log-level", type=str, default='info', choices=['debug', 'info', 'warning', 'error', 'critical'], help="the logging level to stdout")
self.add_argument("--debug", "--verbose", action="store_true", help="set the logging level to debug/verbose mode")

args = parser.parse_args()

print(args)

prompts = load_prompts(args.prompts)

# load model
model = CLIPModel.from_pretrained(
    args.model,
    crop=args.scaling,
    use_tensorrt=(args.api == 'trt'),
    trt_cache=args.trt_cache
)

print_table(model.vision.config)
        
# get similarity scores
# output = model(prompts)
