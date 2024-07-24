#!/usr/bin/env python3
import os
import json
import time
import psutil
import logging
import traceback

import torch
import torch2trt
import tensorrt


from packaging.version import Version
from transformers import CLIPTextModel as CLIPTextModelHF, CLIPTextModelWithProjection, SiglipTextModel, AutoTokenizer
from .utils import AttributeDict, convert_tensor, clip_model_type, trt_model_filename


_clip_text_models = {}


class CLIPTextModel():
    """
    CLIP/SigLIP text encoder and tokenizer for generating text embeddings with TensorRT.
    """
    ModelCache = {}
    
    @staticmethod
    def from_pretrained(model="openai/clip-vit-large-patch14-336", dtype=torch.float16, 
                        projector=None, use_cache=True, use_tensorrt=True, **kwargs):
        """
        Load a CLIP or SigLIP text encoder model from HuggingFace Hub or a local checkpoint.
        Will use TensorRT for inference if ``use_tensorrt=True``, otherwise falls back to Transformers.
        """                
        if use_cache and model in CLIPTextModel.ModelCache:
            return CLIPTextModel.ModelCache[model]
            
        instance = CLIPTextModel(model, dtype=dtype, projector=projector, use_tensorrt=use_tensorrt, **kwargs)
        
        if use_cache:
            CLIPTextModel.ModelCache[model] = instance
            
        return instance
    
    def __init__(self, model, dtype=torch.float16, projector=None, use_tensorrt=True, **kwargs):
        model_types = {
            'clip':  dict(model=CLIPTextModelWithProjection if projector or projector is None else CLIPTextModelHF),
            'siglip': dict(model=SiglipTextModel),
        }
        
        model_type = clip_model_type(model, types=model_types.keys())
        
        if model_type is None:
            raise ValueError(f"tried loading unrecognized CLIP model from {model} - supported model types are CLIP and SigLIP")
        
        if projector is None:
            projector = (model_type == 'clip')

        if model_type == 'siglip':
            if projector:
                projector = False
                logging.warning("disabling projector for SigLIP model {model}")

        self.config = AttributeDict(name=model, type=model_type, projector=projector)
        self.stats = AttributeDict()
        
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.stream = None
        
        use_tensorrt = False  # TEMP
        
        self.dtype = torch.float32 if use_tensorrt else dtype # TRT handles FP16 internally
        self.output_dtype = dtype  # still output the embeddings with the requested dtype
        self.embed_cache = {}
        
        logging.info(f'loading {model_type} text model {model}')

        factory = model_types[model_type]
        
        self.model = factory['model'].from_pretrained(model, torch_dtype=self.dtype)#.to(self.device).eval()

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True, trust_remote_code=True)
        except:
            self.tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False, trust_remote_code=True)

        class TextEncoder(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
                self.config = model.config

            def forward(self, input_ids, attention_mask=None, position_ids=None):
                return self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask, 
                    position_ids=position_ids,
                    output_attentions=False, 
                    output_hidden_states=False, 
                    return_dict=True
                )
         
        self.model = TextEncoder(self.model)
        self.model.to(dtype=self.dtype, device=self.device).eval()
        
        logging.debug(f"{model_type} text model {model}\n\n{self.model}")
        logging.debug(f"{self.config.type} text model warmup ({self.config.name})")
        
        self.embed_text("A dog and a cat sitting on a couch")

        if use_tensorrt:
            try:
                self.init_trt(**kwargs)
            except Exception as error:
                logging.error(f"Exception occurred trying to use TensorRT for {model_type} model ({self.config.name})\n\n{traceback.format_exc()}")

        logging.info(f"loaded {model_type} text model {model}")
        
    def init_trt(self, trt_cache="~/.cache/clip_trt", **kwargs): 
        if Version(tensorrt.__version__) < Version('8.6'):
            logging.warning(f"disabling CLIP with TensorRT {tensorrt.__version__} (requires TensorRT 8.6 or newer)")
            return
            
        if psutil.virtual_memory().total < 20 * (1024 ** 3):
            logging.warning(f"disabling CLIP with TensorRT due to limited memory (falling back to Transformers API)")
            return

        suffix = f"text{'_projector' if self.config.projector else ''}"
        trt_path = os.path.join(os.path.expanduser(trt_cache), trt_model_filename(self.config.name, suffix=suffix))
        #input_ids = torch.ones(*self.config.input_shape, dtype=torch.int64, device='cuda')
        #attention_mask = input_ids.detach().clone()
        input_ids, attention_mask = self.tokenize("A dog and a cat went on a walk around a very long path, it was frought with danger and had many big things in it including far-away items of interest like rocks, germanium, rainbows, and clientel.  A chicken crossed the road to get where he was going. A hippo is a friendly and curious creature.", device=self.device)
        
        if os.path.isfile(trt_path):
            logging.info(f"loading TensorRT model from {trt_path}")
            trt_model = torch2trt.TRTModule()
            trt_model.load_state_dict(torch.load(trt_path))
        else:
            logging.info(f"optimizing {self.config.name} with TensorRT...")
        
            trt_model = torch2trt.torch2trt(
                self.model,
                [input_ids, attention_mask],
                fp16_mode=True,
                log_level=tensorrt.Logger.VERBOSE,
                max_workspace_size=(1024**3) * 3,
                use_onnx=True,
            )
        
            logging.info(f"saving TensorRT model for {self.config.name} to {trt_path}")
            
            os.makedirs(trt_cache, exist_ok=True)
            torch.save(trt_model.state_dict(), trt_path)

        def profile_model(model, runs=3):
            for i in range(runs+1):
                if i == 1:
                    time_begin = time.perf_counter()
                output = model(input_ids, attention_mask)
            torch.cuda.synchronize()
            return (time.perf_counter() - time_begin) * 1000 / runs

        key = 'text_embeds' if self.config.projector else 'pooler_output'

        logging.info(f"benchmarking {self.config.type} text model {self.config.name}")
        logging.info(f"torch time:  {profile_model(self.model)} ms")
        logging.info(f"trt time:    {profile_model(trt_model)} ms")
        logging.info(f"y^ delta:    {torch.max(torch.abs(self.model(input_ids, attention_mask)[key] - trt_model(input_ids, attention_mask)[key]))}")
          
        trt_model.config = self.model.config
        
        self.model = trt_model
        self.embed_text("A dog and a cat sitting on a couch")
     
    def tokenize(self, text, padding='max_length', truncation=True, dtype=torch.int64, return_tensors='pt', return_dict=False, device=None, **kwargs):
        """
        Tokenize the given string and return the encoded token ID's and attention mask (either in a dict or as a tuple).
        
        Args:
          text (str): the text to tokenize.
          dtype (type): the numpy or torch datatype of the tensor to return.
          return_tensors (str): ``'np'`` to return a `np.ndarray` or ``'pt'`` to return a `torch.Tensor`
          kwargs:  additional arguments forwarded to the HuggingFace `transformers.AutoTokenizer <https://huggingface.co/docs/transformers/main/en/model_doc/auto#transformers.AutoTokenizer>`_ encode function.
          
        Returns:
          The token ID's with the tensor type as indicated by `return_tensors` (either `'np'` for `np.ndarray`
          or `'pt'` for `torch.Tensor`) and datatype as indicated by `dtype` (by default ``int65``)
        """
        output = self.tokenizer(
            text, 
            padding=padding, 
            truncation=truncation,
            return_tensors=return_tensors,
            return_attention_mask=True,
            **kwargs
        )

        output.input_ids = convert_tensor(output.input_ids, return_tensors=return_tensors, dtype=dtype, device=device)
        output.attention_mask = convert_tensor(output.attention_mask, return_tensors=return_tensors, dtype=dtype, device=device)

        if return_dict:
            return output
        else:
            return output.input_ids, output.attention_mask

    def embed_tokens(self, tokens, attention_mask=None, return_tensors='pt', stream=None, **kwargs):   
        """
        Return the embedding features of the given tokens. The attention mask is typically used as these models use padding.
        """
        with torch.cuda.StreamContext(stream), torch.inference_mode():
            time_begin_enc = time.perf_counter()
            
            tokens = convert_tensor(tokens, return_tensors='pt', device=self.device)
            attention_mask = convert_tensor(attention_mask, return_tensors='pt', device=self.device)
            
            if len(tokens.shape) == 1:
                tokens = tokens.unsqueeze(0)
             
            if attention_mask is not None and len(attention_mask.shape) == 1:
                attention_mask = attention_mask.unsqueeze(0)
                   
            output = self.model(tokens, attention_mask)
            output = output['text_embeds' if self.config.projector else 'pooler_output']
            output = convert_tensor(output, return_tensors=return_tensors, device=self.device, dtype=self.output_dtype)
            
            self.config.input_shape = tokens.shape
            self.config.output_shape = output.shape
 
        time_end_enc = time.perf_counter()
        
        self.stats.time = time_end_enc - time_begin_enc
        self.stats.rate = 1.0 / self.stats.time
        self.stats.input_shape = self.config.input_shape
        self.stats.output_shape = self.config.output_shape

        return output

    def embed_text(self, text, use_cache=False, **kwargs):
        """
        Return the embedding features of the given text.
        """
        output = None
        
        if use_cache:
            output = self.embed_cache.get(text)
            logging.debug(f"{self.config.type} text embedding cache hit `{text}`".replace('\n', '\\n'))
            
        if output is None:
            tokens, attention_mask = self.tokenize(text, **kwargs)
            output = self.embed_tokens(tokens, attention_mask=attention_mask, **kwargs)
            if use_cache:
                self.embed_cache[text] = output
            
        return output
            
    def __call__(self, text, **kwargs):
        if text is None:
            return
        elif isinstance(text, str):
            return self.embed_text(text, **kwargs)
        else:
            return self.embed_tokens(text, **kwargs)
            
        
