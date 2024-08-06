import os
import PIL
import json
import time
import psutil
import pprint
import logging
import functools
import safetensors

import timm
import torch
import torchvision

import tensorrt
import torch2trt

from clip_trt.utils import torch_image, load_image, image_size, convert_tensor, trt_model_filename, AttributeDict
from packaging.version import Version


class TIMMVisionModel():
    """
    TIMM vision transformers accelerated with TensorRT (using torch2trt)
    """
    ModelCache = {}
    
    @staticmethod
    def from_pretrained(model="vit_large_patch14_reg4_dinov2.lvd142m", use_cache=True, **kwargs):
        """
        Load a TIMM vision encoder model from HuggingFace Hub, torch.hub, or a local checkpoint.
        Will use TensorRT for inference if ``use_tensorrt=True``, otherwise falls back to PyTorch.
        The kwargs are passed to :func:`timm2trt`, see that function for the available model options.
        """                
        if use_cache:
            model_config = frozenset({'model': model, **kwargs}.items())
            cached_model = TIMMVisionModel.ModelCache.get(model_config)
            if cached_model is not None:
                return cached_model
            
        instance = TIMMVisionModel(model, **kwargs)
        
        if use_cache:
            TIMMVisionModel.ModelCache[model_config] = instance
            
        return instance
    
    @staticmethod
    def list_models(self, module='vision_transformer', **kwargs):
        """
        List the available TIMM models - see here for kwargs to ``timm.list_models()`` for filtering:
        
        https://timm.fast.ai/#List-Models-with-Pretrained-Weights
        https://github.com/huggingface/pytorch-image-models/blob/20fe56bd9072af61d9f5404ce8b08e24ff10a807/timm/models/_registry.py#L185
        """
        return timm.list_models(module=module, **kwargs)
        
    def __init__(self, model, dtype=torch.float16, device='cuda:0', **kwargs):
        """
        Initialize a TIMM model instance (see :func:`timm2trt` for kwargs)
        """
        self.dtype, self.device = dtype, device
        self.model, self.preprocessor = timm2trt(model, **kwargs)
        
        img_size = self.preprocessor.transforms[0].size
        
        self.config = AttributeDict(name=model, hidden_state=kwargs.get('hidden_state'), input_shape=(img_size, img_size))
        self.stats = AttributeDict()
        
        self(PIL.Image.new('RGB', self.config.input_shape, (255,255,255)))
        
        logging.info(f"loaded TIMM vision model {model}  (use_tensorrt={isinstance(self.model, torch2trt.TRTModule)})")
        
    def embed_image(self, image, return_tensors='pt', stream=None, **kwargs):
        """
        Return the encoded feature embeddings from the given image (or whatever the model output is).
        """   
        if isinstance(image, str):
            image = load_image(image)
            
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
            
            output = self.model(image)
            output = convert_tensor(output, return_tensors=return_tensors, device=self.device, dtype=self.dtype)
            
            self.config.output_shape = tuple(output.shape)
            
        #torch.cuda.synchronize()
        time_end_enc = time.perf_counter()
        
        self.stats.time = time_end_enc - time_begin_enc
        self.stats.rate = 1.0 / self.stats.time
        self.stats.input_shape = f"{tuple(image_size(image))} -> {self.config.input_shape}"
        self.stats.output_shape = self.config.output_shape

        return output
        
    def __call__(self, image, **kwargs):
        return self.embed_image(image, **kwargs)
        
        
def timm2trt(model="vit_large_patch14_reg4_dinov2.lvd142m", 
             weights=None, weights_key=None, 
             dtype=torch.float16, device='cuda:0', 
             transform={}, hidden_state=None, benchmark_runs=25, 
             use_tensorrt=True, trt_cache="~/.cache/clip_trt", 
             **kwargs):
    """
    Load a TIMM vision model with TensorRT, performing the conversion if needed with torch2trt.
    If a directory of weights is provided, they will be loaded into the model instead of the originals.
    The kwargs are passed to ``timm.create_model()`` (https://huggingface.co/docs/timm/en/reference/models)
    Returns a ``(model, preprocessor)`` tuple of ``torch.nn.Module`` (see :class:`TIMMVisionModel` for a wrapper)
    """
    if Version(tensorrt.__version__) < Version('8.6'):
        logging.warning(f"disabling TIMM vision models with TensorRT {tensorrt.__version__} (requires TensorRT 8.6 or newer)")
        use_tensorrt = False
        
    if psutil.virtual_memory().total < 20 * (1024 ** 3):
        logging.warning(f"disabling CLIP with TensorRT due to limited memory (falling back to PyTorch)")
        use_tensorrt = False
            
    logging.info(f"loading TIMM vision model '{model}' (use_tensorrt={use_tensorrt}, {dtype})")

    trt_suffix = []
    
    if hidden_state is not None:
        trt_suffix.append(f"hidden{hidden_state}")
     
    if 'img_size' in kwargs:
        transform['img_size'] = kwargs['img_size']
        
    for key, value in transform.items():
        trt_suffix.append(f"{key}={value}")
 
    timm_config = timm.get_pretrained_cfg(model)
    logging.info(f"TIMM model '{model}' configuration:\n\n{pprint.pformat(timm_config, indent=2)}")
     
    trt_path = os.path.join(os.path.expanduser(trt_cache), trt_model_filename(model, suffix='_'.join(trt_suffix)))
    trt_found = os.path.isfile(trt_path)

    timm_model = timm.create_model(
        model,
        pretrained=bool(not weights),
        exportable=use_tensorrt,
        **kwargs,
    ) if not use_tensorrt or not trt_found else None

    if timm_model:
        if weights:
            with open(os.path.join(weights, 'model.safetensors.index.json')) as file:
                weight_map = json.load(file)['weight_map']
                
            weight_files = {}
            weight_tensors = {}
            
            for layer_name, weight_file in weight_map.items():
                weight_files[weight_file] = weight_files.get(weight_file, []) + [layer_name]
            
            for weight_file, layers in weight_files.items():
                weight_path = os.path.join(weights, weight_file)
                logging.info(f"TIMM model '{model}' loading {weight_path}")   
                      
                with safetensors.safe_open(weight_path, framework='pt', device='cpu') as file:
                    for layer_name in layers:
                        if weights_key:
                            renamed_layer = weights_key(layer_name)
                        else:
                            renamed_layer = layer_name
                            
                        if not renamed_layer:
                            continue
                        
                        #logging.debug(f"loading {model} layer weights {layer_name} as {renamed_layer}")  
                        weight_tensors[renamed_layer] = file.get_tensor(layer_name)

            timm_model.load_state_dict(weight_tensors)     

        def unpack_tuple(fn):
            def wrapper(*args, **kwargs):
                result = fn(*args, **kwargs)
                return result[0] if isinstance(result, (tuple,list)) else result
            return wrapper

        if hidden_state is not None:
            if hidden_state < 0:
                hidden_state = len(timm_model.blocks) + hidden_state
            timm_model.forward = unpack_tuple(
                functools.partial(timm_model.get_intermediate_layers, n={hidden_state})
            )
            logging.debug(f"TIMM model '{model}' output hidden state {hidden_state}")
        
        timm_model = timm_model.to(dtype=dtype, device=device).eval()

    # get model specific transforms (normalization, resize)   
    data_config = timm.data.resolve_model_data_config(timm_model, args=transform, pretrained_cfg=timm_config.to_dict())
    transforms = timm.data.create_transform(**data_config, is_training=False)
    logging.debug(f"TIMM model '{model}' preprocessing transforms:\n{data_config}\n{transforms}")

    # remove ToTensor() because TIMM assumes PIL.Image inputs
    i=0
    while i < len(transforms.transforms):
        if isinstance(transforms.transforms[i], torchvision.transforms.ToTensor):
            del transforms.transforms[i]
        else:
            i += 1
    
    # create dummy input and apply pre-processing transforms
    input = torch_image(PIL.Image.new('RGB', (512,512), (255,255,255)), dtype=dtype, device=device)
    input = transforms(input).unsqueeze(0)
    
    if timm_model:
        output = timm_model(input)
        logging.debug(f"TIMM model '{model}' inputs:  shape={input.shape}  dtype={input.dtype}  device={input.device}")
        logging.debug(f"TIMM model '{model}' output:  shape={output.shape}  dtype={output.dtype}  device={output.device}")
        
    # load TensorRT model (or build it first)
    if not use_tensorrt:
        return timm_model, transforms
        
    trt_path = os.path.join(os.path.expanduser(trt_cache), trt_model_filename(model, suffix='_'.join(trt_suffix)))

    if trt_found:
        logging.info(f"loading TensorRT engine for TIMM model '{model}' from {trt_path}")
        trt_model = torch2trt.TRTModule()
        trt_model.load_state_dict(torch.load(trt_path))
    else:
        logging.info(f"optimizing TIMM model '{model}' with TensorRT...")
    
        trt_model = torch2trt.torch2trt(
            timm_model,
            [input],
            fp16_mode=(dtype == torch.float16),
            log_level=tensorrt.Logger.VERBOSE,
            max_workspace_size=(1024**3) * 3,
            use_onnx=True,
        )
    
        logging.info(f"saving TensorRT model for {model} to {trt_path}")
        
        os.makedirs(trt_cache, exist_ok=True)
        torch.save(trt_model.state_dict(), trt_path)
                
    # run benchmarking and validation
    def profile_model(model, input, runs=3):
        for i in range(runs+1):
            if i == 1:
                time_begin = time.perf_counter()
            output = model(input)
        torch.cuda.synchronize()
        return (time.perf_counter() - time_begin) * 1000 / runs
        
    logging.info(f"benchmarking TIMM vision model {model}")
    logging.info(f"trt_time:    {profile_model(trt_model, input, runs=benchmark_runs)} ms")
    
    if timm_model:
        logging.info(f"torch_time:  {profile_model(timm_model, input, runs=benchmark_runs)} ms")
        logging.info(f"y^ RMSE:     {torch.sqrt(torch.mean(torch.pow(torch.abs(trt_model(input) - timm_model(input)),2.0)))}")   #{torch.max(torch.abs(timm_model(input) - trt_model(input)))}")
        #trt_model.embed_dim = timm_model.embed_dim
        del timm_model
        
    return trt_model, transforms


if __name__ == '__main__':
    import sys
    import urllib
    import argparse
    
    from clip_trt.utils import LogFormatter, print_table

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model', type=str, default="vit_large_patch14_reg4_dinov2.lvd142m")  # 'vit_large_patch14_dinov2.lvd142m'  # embedding model
    parser.add_argument('--weights', type=str, default=None)
    parser.add_argument('--weights-prefix', type=str, default=None)
    parser.add_argument('--hidden-state', type=int, default=None)
    parser.add_argument('--img-size', type=int, default=None)
    parser.add_argument('--list-models', action='store_true')
    parser.add_argument('--api', type=str, default='TIMMVisionModel', choices=['TIMMVisionModel', 'timm2trt'])
    
    args = parser.parse_args()
    LogFormatter.config(level='debug')
    logging.debug(args)
    
    if args.list_models:
        pprint.pprint(TIMMVisionModel.list_models(), indent=2)
        sys.exit(0)
        
    if args.api == 'TIMMVisionModel':
        model = TIMMVisionModel.from_pretrained(
            args.model, 
            weights=args.weights, 
            weights_key=lambda layer: layer.replace(args.weights_prefix, '').replace('scale_factor', 'gamma') if args.weights_prefix in layer else None,
            hidden_state=args.hidden_state,
            img_size=args.img_size,
            num_classes=0,
            act_layer=None,
        )
        
        print_table(model.config)
        img = load_image("/data/images/dogs.jpg")
        
        for i in range(1):
            output = model(img)
            #print_table(model.stats)
          
        print("")
        print(f"inputs:  shape={tuple(image_size(img))}  type={type(img)}")
        print(f"output:  shape={output.shape}  dtype={output.dtype}  device={output.device}")
         
    else:
        model, transforms = timm2trt(
            args.model, 
            weights=args.weights, 
            weights_key=lambda layer: layer.replace(args.weights_prefix, '').replace('scale_factor', 'gamma') if args.weights_prefix in layer else None,
            hidden_state=args.hidden_state,
            img_size=args.img_size,
            num_classes=0,
            act_layer=None,
        )
        
        img = Image.open(urllib.request.urlopen(
            "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png"
        ))
        
        img = torch_image(img, dtype=torch.float16, device='cuda:0')
        input = transforms(img).unsqueeze(0)
        
        output = model(input)
        
        print("")
        print(f"inputs:  shape={input.shape}  dtype={input.dtype}  device={input.device}")
        print(f"output:  shape={output.shape}  dtype={output.dtype}  device={output.device}")
        #print(f"embeds:  {model.embed_dim}")
    
    # classify
    #top5_probabilities, top5_class_indices = torch.topk(output.softmax(dim=1) * 100, k=5)
    #print('\nPROBS', top5_probabilities, '\nINDICES', top5_class_indices)

    # or equivalently (without needing to set num_classes=0)
    #output = model.forward_features(transforms(img).unsqueeze(0))
    # output is unpooled, a (1, 1370, 1024) shaped tensor

    #output = model.forward_head(output, pre_logits=True)
    # output is a (1, num_features) shaped tensor

