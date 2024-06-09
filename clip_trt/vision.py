#!/usr/bin/env python3
import os
import json
import time
import psutil
import logging
import traceback

import PIL
import torch
import torch2trt
import tensorrt

import torchvision.transforms as T

from packaging.version import Version
from transformers import CLIPVisionModel as CLIPVisionModelHF, CLIPVisionModelWithProjection, SiglipVisionModel
from .utils import AttributeDict, load_image, torch_image, image_size, convert_tensor, clip_model_type, trt_model_filename


_clip_vision_models = {}


class CLIPVisionModel():
    """
    CLIP/SigLIP vision encoder for generating image embeddings with TensorRT.
    """
    @staticmethod
    def from_pretrained(model="openai/clip-vit-large-patch14-336", dtype=torch.float16, 
                        projector=None, crop=None, use_cache=True, use_tensorrt=True, **kwargs):
        """
        Load a CLIP or SigLIP vision encoder model from HuggingFace Hub or a local checkpoint.
        Will use TensorRT for inference if ``use_tensorrt=True``, otherwise falls back to Transformers.
        """                
        global _clip_vision_models
        
        if use_cache and model in _clip_vision_models:
            return _clip_vision_models[model]
            
        instance = CLIPVisionModel(model, dtype=dtype, projector=projector, crop=crop, use_tensorrt=use_tensorrt, **kwargs)
        
        if use_cache:
            _clip_vision_models[model] = instance
            
        return instance
    
    def __init__(self, model, dtype=torch.float16, projector=None, crop=None, use_tensorrt=True, **kwargs):
        clip_class = CLIPVisionModelWithProjection if projector or projector is None else CLIPVisionModelHF
        
        model_types = {
            'clip':  dict(model=clip_class, mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)),
            'siglip': dict(model=SiglipVisionModel, mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)),
        }
        
        model_type = clip_model_type(model, types=model_types.keys())
        
        if model_type is None:
            raise ValueError(f"tried loading unrecognized CLIP model from {model} - supported model types are CLIP and SigLIP")
        
        if projector is None:
            projector = (model_type == 'clip')
            
        if crop is None:
            crop = (model_type == 'clip')
        elif isinstance(crop, str):
            crop = (crop == 'crop')
                
        if model_type == 'siglip':
            if projector:
                projector = False
                logging.warning("disabling projector for SigLIP model {model}")
            if crop:
                logging.warning("SigLIP models don't typically have cropping enabled")
                
        self.config = AttributeDict(name=model, type=model_type, projector=projector, crop=crop)
        self.stats = AttributeDict()
        
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.stream = None
        
        self.dtype = torch.float32 if use_tensorrt else dtype # TRT handles FP16 internally
        self.output_dtype = dtype  # still output the embeddings with the requested dtype

        logging.info(f'loading {model_type} vision model {model}')

        factory = model_types[model_type]
        
        self.model = factory['model'].from_pretrained(model, torch_dtype=self.dtype)#.to(self.device).eval()
        self.config.input_shape = (self.model.config.image_size, self.model.config.image_size)
        
        #self.preprocessor = model_type['preprocessor'].from_pretrained(model, torch_dtype=self.dtype)#.to(self.device)
        
        # Pre-processing is able to use GPU with torchvision (cropping is optional)
        # https://github.com/openai/CLIP/blob/a1d071733d7111c9c014f024669f959182114e33/clip/clip.py#L79
        self.preprocessor = torch.nn.Sequential()

        self.preprocessor.append(
            T.Resize(
                self.config.input_shape[0] if crop else self.config.input_shape, 
                interpolation=T.InterpolationMode.BICUBIC# BILINEAR
            )
        )
        
        if crop:
            self.preprocessor.append(T.CenterCrop(self.config.input_shape[0]))
   
        self.preprocessor.append(T.Normalize(factory['mean'], factory['std']))
        self.preprocessor.append(T.ConvertImageDtype(self.dtype))

        class VisionEncoder(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
                self.config = model.config

            def forward(self, image):
                return self.model(
                    image, 
                    output_attentions=False, 
                    output_hidden_states=True, 
                    return_dict=True
                )
         
        self.model = VisionEncoder(self.model)
        self.model.to(dtype=self.dtype, device=self.device).eval()
        
        logging.debug(f"{model_type} vision model {model}\n\n{self.model}")
        logging.debug(f"{model_type} vision model warmup ({model})")
        
        self(PIL.Image.new('RGB', self.config.input_shape, (255,255,255)))

        if use_tensorrt:
            try:
                self.init_trt(**kwargs)
            except Exception as error:
                logging.error(f"Exception occurred trying to use TensorRT for {model_type} model ({self.config.name})\n\n{traceback.format_exc()}")

        logging.success(f"loaded {model_type} vision model {model}")
        
    def init_trt(self, trt_cache="~/.cache/clip_trt", **kwargs): 
        if Version(tensorrt.__version__) < Version('8.6'):
            logging.warning(f"disabling CLIP with TensorRT {tensorrt.__version__} (requires TensorRT 8.6 or newer)")
            return
            
        if psutil.virtual_memory().total < 20 * (1024 ** 3):
            logging.warning(f"disabling CLIP with TensorRT due to limited memory (falling back to Transformers API)")
            return

        suffix = f"vision{'_projector' if self.config.projector else ''}"
        trt_path = os.path.join(os.path.expanduser(trt_cache), trt_model_filename(self.config.name, suffix=suffix))
        test_inputs = torch.ones(1, 3, *self.config.input_shape, dtype=self.dtype, device='cuda')

        if os.path.isfile(trt_path):
            logging.info(f"loading TensorRT model from {trt_path}")
            trt_model = torch2trt.TRTModule()
            trt_model.load_state_dict(torch.load(trt_path))
        else:
            logging.info(f"optimizing {self.config.name} with TensorRT...")
        
            trt_model = torch2trt.torch2trt(
                self.model,
                [test_inputs],
                fp16_mode=True,#(self.config.dtype == torch.float16),
                log_level=tensorrt.Logger.VERBOSE,
                max_workspace_size=(1024**3) * 3,
                use_onnx=True,
            )
        
            logging.info(f"saving TensorRT model for {self.config.name} to {trt_path}")
            
            os.makedirs(trt_cache, exist_ok=True)
            torch.save(trt_model.state_dict(), trt_path)
        
        def profile_model(model, inputs, runs=3):
            for i in range(runs+1):
                if i == 1:
                    time_begin = time.perf_counter()
                output = model(inputs)
            torch.cuda.synchronize()
            return (time.perf_counter() - time_begin) * 1000 / runs
            
        key = 'image_embeds' if self.config.projector else 'pooler_output'
        
        logging.info(f"benchmarking {self.config.type} vision model {self.config.name}")
        logging.info(f"torch time:  {profile_model(self.model, test_inputs)} ms")
        logging.info(f"trt time:    {profile_model(trt_model, test_inputs)} ms")
        logging.info(f"y^ delta:    {torch.max(torch.abs(self.model(test_inputs)[key] - trt_model(test_inputs)[key]))}")
          
        trt_model.config = self.model.config
        self.model = trt_model
        
    def embed_image(self, image, hidden_state=None, return_tensors='pt', return_dict=False, stream=None, **kwargs):
        """
        Return the encoded features from the given image in the embedding (or whatever the model output is).
        """
        if isinstance(image, str):
            image = load_image(image)
        
        def _convert_tensor(x):
            return convert_tensor(x, return_tensors=return_tensors, device=self.device, dtype=self.output_dtype)
            
        output = AttributeDict() if return_dict else None
        
        with torch.cuda.StreamContext(stream), torch.inference_mode():
            time_begin_enc = time.perf_counter()
            
            image = torch_image(image, dtype=self.dtype, device=self.device)
            ndims = len(image.shape)

            if ndims != 3 and ndims != 4:
                raise ValueError(f"image with dims {image.shape} was not in NCHW or NHWC format")
            
            if ndims == 3:
                image = image.unsqueeze(0)
                
            if image.shape[3] <= 4:
                image = image.permute(0, 3, 1, 2)
                
            image = self.preprocessor(image)
            model_output = self.model(image) #, output_hidden_states=hidden_state is not None)   #.pooler_output  .last_hidden_state
            output_embeds = model_output['image_embeds' if self.config.projector else 'pooler_output']
  
            if hidden_state is not None:
                hidden_tensor = _convert_tensor(model_output['hidden_states'][hidden_state])
                if return_dict:
                    output.hidden_state = hidden_tensor
                else:
                    output = hidden_tensor
                self.config.output_shape = hidden_tensor.shape
            else:
                self.config.output_shape = output_embeds.shape
                
            if return_dict:
                output.image_embeds = _convert_tensor(output_embeds) 
            elif hidden_state is None:
                output = _convert_tensor(output_embeds) 

        time_end_enc = time.perf_counter()
        
        self.stats.time = time_end_enc - time_begin_enc
        self.stats.rate = 1.0 / self.stats.time
        self.stats.input_shape = f"{image_size(image)} -> {self.config.input_shape}"
        self.stats.output_shape = self.config.output_shape

        return output
        
    def __call__(self, image, **kwargs):
        return self.embed_image(image, **kwargs)
        
