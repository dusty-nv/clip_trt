#!/usr/bin/env python3
import torch
import transformers
import logging

from .text import CLIPTextModel
from .vision import CLIPVisionModel

from .utils import ImageExtensions, is_image, is_embedding


class CLIPModel():
    """
    CLIP/SigLIP text + vision encoders for generating similar feature embeddings.
    """
    @staticmethod
    def from_pretrained(model="openai/clip-vit-large-patch14-336", text=True, vision=True, **kwargs):
        """
        Load CLIP or SigLIP text and vision models from HuggingFace Hub or a local checkpoint.
        Will use TensorRT for inference if ``use_tensorrt=True``, otherwise falls back to Transformers.
        """         
        return CLIPModel(model, text=text, vision=vision, **kwargs)       
    
    def __init__(self, model, text=True, vision=True, **kwargs):
        """
        Load CLIP or SigLIP text and vision models from HuggingFace Hub or a local checkpoint.
        Will use TensorRT for inference if ``use_tensorrt=True``, otherwise falls back to Transformers.
        """    
        auto_model = transformers.AutoModel.from_pretrained(model)

        self.logit_scale = auto_model.logit_scale.detach().clone().cuda()
        self.logit_scale_exp = self.logit_scale.exp()
        
        if hasattr(auto_model, 'logit_bias'):
            self.logit_bias = auto_model.logit_bias.detach().clone().cuda()
        else:
            self.logit_bias = torch.tensor(0, dtype=torch.float32, device='cuda')
            
        del auto_model
        
        #: The CLIPTextModel (loaded when ``text=True``)
        self.text = CLIPTextModel.from_pretrained(model, **kwargs) if text else None
        
        #: The CLIPVisionModel (loaded when ``vision=True``)
        self.vision = CLIPVisionModel.from_pretrained(model, **kwargs) if vision else None
        
        #: The name or path of the model that was loaded.
        self.model_name = model
 
        logging.debug(f"{self.model_name}  logit_scale={self.logit_scale}  logit_scale_exp={self.logit_scale_exp}  logit_bias={self.logit_bias}")

    def embed_image(self, image, **kwargs):
        """
        Return the encoded feature embedding for the given image(s).
        The model should have been loaded with ``vision=True`` to use this.
        See :func:`CLIPVisionModel.embed_image` for the kwargs. 
        """
        if self.vision is None:
            raise RuntimeError(f"{self.model_name} should have been loaded with ``vision=True`` to use CLIPModel.embed_image()")
        
        return self.vision.embed_image(image, **kwargs)
     
    def embed_text(self, text, **kwargs):
        """
        Return the encoded feature embedding for the given text.
        The model should have been loaded with ``text=True`` to use this.
        See :func:`CLIPTextModel.embed_image` for the kwargs. 
        """
        if self.text is None:
            raise RuntimeError(f"{self.model_name} should have been loaded with ``text=True`` to use CLIPModel.embed_text()")

        return self.text.embed_text(text, **kwargs)
    
    def embed(self, inputs, normalize=True, **kwargs):
        """
        Embed a list of text or image inputs
        """
        if inputs is None:
            return None
            
        if is_embedding(inputs):
            return inputs
            
        def _embed(input, **kwargs):
            if input is None:
                return None
            elif is_embedding(input):
                return input
            elif is_image(input) or (isinstance(input, str) and (input.endswith(ImageExtensions) or input.startswith('http'))):
                return self.embed_image(input, **kwargs)
            elif isinstance(input, str):
                return self.embed_text(input, **kwargs)
            else:
                raise TypeError(f"CLIPModel.embed() expects inputs of type str, torch.Tensor, or np.ndarray (was {type(input)})")
          
        if isinstance(inputs, list):
            embeds = torch.cat([_embed(input, **kwargs) for input in inputs], dim=0)
        else:
            embeds = _embed(inputs, **kwargs)

        if normalize:
            embeds /= embeds.norm(p=2, dim=-1, keepdim=True)

        return embeds
        
    def similarity(self, *args, normalize=True, logit_scaling=False, return_probs=False, **kwargs):
        """
        Compute the similarity between groups of inputs.  These can be embeddings or lists of text, images, or image filenames.
        """
        num_inputs = len(args)
        
        if num_inputs == 1:
            x = self.embed(args[0], normalize=normalize, **kwargs)
            y = x
        elif num_inputs == 2:
            x = self.embed(args[0], normalize=normalize, **kwargs)
            y = self.embed(args[1], normalize=normalize, **kwargs)   
        else:
            raise ValueError(f"CLIPModel.similarity() expects 1 or 2 input groups (got {num_inputs})")
            
        output = torch.matmul(x, y.t())
        
        if logit_scaling:
            output = output * self.logit_scale_exp + self.logit_bias

        if return_probs:
            if self.vision.config.type == 'clip':
                output = torch.softmax(output, dim=-1)
            elif self.vision.config.type == 'siglip':
                output = torch.sigmoid(output)  # 1/(1 + torch.exp(-output))

        return output
     
    def __call__(self, *args, **kwargs):
        """
        Compute the similarity between groups of inputs.  These can be embeddings or lists of text, images, or image filenames.
        """
        return self.similarity(*args, **kwargs)


