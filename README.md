# clip_trt
CLIP and SigLIP models optimized with TensorRT with a Transformers-like API

### Command-Line Example

```bash
python3 -m clip_trt \
  --inputs image_a.jpg image_b.jpg image_c.jpg \
  --inputs 'a dog' 'a cat' 'a bear' 'a lion'
```

### Code Example

```python
from clip_trt import CLIPModel

model = CLIPModel.from_pretrained(
    "openai/clip-vit-large-patch14-336",
    use_tensorrt=True,
    crop=False,
)

similarity = model(
    [
        'my_image.jpg',
        PIL.Image.open('image_2.jpg').convert('RGB'),
        np.ndarray((3,480,640), dtype=np.uint8),
        torch.ones((3,336,336), dtype=torch.uint8, device='cuda')
    ],
    [
        'a dog', 'a cat', 'a bear', 'a lion'
    ],
)
```

### Embeddings

```
image_embed = model.embed('xyz.jpg')
text_embed = model.embed('an elephant in the jungle')
```
