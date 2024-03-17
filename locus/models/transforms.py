import torch
from torchvision.transforms import v2

LDoGItransforms = v2.Compose(
    [
        v2.ToImage(),  # Convert to tensor, only needed if you had a PIL image
        # ...
        v2.ToDtype(torch.float32, scale=True),  # Normalize expects float input
        v2.Resize(224, antialias=True),
        v2.RandomCrop(size=(224)),  # Or Resize(antialias=True)
        # ...
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # v2.ToDtype(torch.uint8, scale=True),
    ]
)
