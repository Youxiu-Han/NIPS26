import torch
from torch import nn
import torch.nn.functional as F

img = torch.randn(96, 3, 224, 224)
patch_size = 16
prompt = torch.randn(1, 3, 16, 16)

image_patch = F.unfold(img, (patch_size, patch_size), stride=patch_size)
print('image_patch', image_patch.shape)
indices = torch.randint(0, 196, (64, 5))
print('indices', indices.shape)
indices_expanded = indices.unsqueeze(1).expand(-1, 768, -1)
print('indices_expanded', indices_expanded.shape)
extracted_patches = torch.gather(image_patch, 2, indices_expanded)
print('extracted_patches', extracted_patches.shape)

