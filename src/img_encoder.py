import torch
import torch.nn as nn
import open_clip
from PIL import Image
import os
import numpy as np

MODEL = 'ViT-B-32'
PRETRAINED = 'laion2b_s34b_b79k'
DIM = 512

class ImgEncoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(MODEL, PRETRAINED)
        self.dimention = DIM

    def forward(self, images_tensor):
        with torch.no_grad():
            embeddings = self.model.encode_image(images_tensor) # embedding the images   
            embeddings /= embeddings.norm(dim=-1, keepdim=True) # normalizing
            return embeddings
                


