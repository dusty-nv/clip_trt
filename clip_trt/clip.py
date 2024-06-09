#!/usr/bin/env python3
from .vision import CLIPVisionModel


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
        #: The CLIPTextModel (loaded when ``text=True``)
        #self.text_model = CLIPTextModel.from_pretrained(model, **kwargs) if text else None
        
        #: The CLIPVisionModel (loaded when ``vision=True``)
        self.vision_model = CLIPVisionModel.from_pretrained(model, **kwargs) if vision else None
        
        #: The name or path of the model that was loaded.
        self.model_name = model

    def embed_image(self, image, **kwargs):
        """
        Return the encoded feature embedding for the given image(s).
        The model should have been loaded with ``vision=True`` to use this.
        See :func:`CLIPVisionModel.embed_image` for the kwargs. 
        """
        if self.vision_model is None:
            raise RuntimeError(f"{self.model_name} should have been loaded with ``vision=True`` to use CLIPModel.embed_image()")
        
        return self.vision_model.embed_image(image, **kwargs)
        
    #def __call__(self, image, hidden_state=None, return_tensors='pt', **kwargs):
    #    return self.embed_image(image, hidden_state=hidden_state, return_tensors='pt', **kwargs)
        
