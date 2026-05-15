import torch
import torch.nn as nn
import open_clip
from PIL import Image
import os
import numpy as np

MODEL = 'ViT-B-32'
PRETRAINED = 'laion2b_s34b_b79k'
DIM = 512

class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model, _, _ = open_clip.create_model_and_transforms(MODEL, PRETRAINED)
        self.tokenizer = open_clip.get_tokenizer(MODEL)
        self.dimention = DIM

    def forward(self, tokens):
        with torch.no_grad():
            embeddings = self.model.encode_text(tokens) # embedding the text
            embeddings /= embeddings.norm(dim=-1, keepdim=True) # normalizing
        return embeddings
    
    