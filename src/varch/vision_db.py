import os
from PIL import Image

import faiss
import numpy as np

import torch
from tqdm import tqdm
from varch.encoder import Encoder

VALID_FORMATS = ('.jpg', '.jpeg', '.png')

class VisionDB():
    def __init__(self, path, device, batch_size=32, hnsw_links=0, load=False):
        self.path = os.path.join(path, "db")
        os.makedirs(self.path, exist_ok=True)
        self.device = device
        self.batch_size = batch_size
        self.encoder = Encoder().to(self.device)
        self.encoder.eval()
        self.paths = []
        self.index = faiss.IndexFlatIP(self.encoder.dimension) if not hnsw_links else faiss.IndexHNSWFlat(self.encoder.dimension, hnsw_links)
        if load:
            self.load()

    @torch.no_grad()    
    def observe(self, path):
        """Observes all images in path and embedds them into the DB"""
        
        assert not self.paths
        print(f"observing: {path}")
        images = [os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith(VALID_FORMATS)]
        
        # working in batches 
        for i in tqdm(range(0, len(images), self.batch_size)):
            batch_paths = images[i : i + self.batch_size]
            batch_images = []
            valid_batch_paths = []

            # preprocessing batch on cpu
            for img_path in batch_paths:
                try:
                    img = Image.open(img_path).convert('RGB')
                    batch_images.append(self.encoder.preprocess(img))
                    valid_batch_paths.append(img_path)
                except Exception as e:
                    print(f"skipping {img_path}: {e}")

            # embedding the images
            if batch_images:
                # stack images into a tensor [B, 3, H, W]
                images_tensor = torch.stack(batch_images).to(self.device)
                
                # embedding the images
                embeddings = self.encoder.embedd_images(images_tensor)
                embeddings_np = embeddings.cpu().numpy().astype('float32')
                
                self.index.add(embeddings_np)
                self.paths.extend(valid_batch_paths)
        
        self.save()
        print(f"indexed {self.index.ntotal}/{len(images)} images")

    @torch.no_grad()
    def search(self, text=None, image_path=None, k=5):
        """Searches the DB"""

        assert self.paths
        assert text or image_path

        query_vec = None
        if text:
            # tokenizing
            tokens = self.encoder.tokenizer(text)
            tokens = tokens.to(self.device)
            
            # embedding the text
            query_vec = self.encoder.embedd_text(tokens)
            query_vec = query_vec.to('cpu')

        elif image_path:
            # embedding the image
            image = Image.open(image_path).convert('RGB')
            image = torch.stack([self.encoder.preprocess(image)]).to(self.device) # 'tensoring' the image
            query_vec = self.encoder.embedd_images(image)
            query_vec = query_vec.to('cpu')

        D, I = self.index.search(query_vec, k)
        relevant_images, scores =  [self.paths[idx] for idx in I[0] if idx != -1], D[0]
        return relevant_images, scores

    def save(self):
        """Saves the DB"""

        print("saving db...")
        faiss.write_index(self.index, os.path.join(self.path, "embeddings.index"))
        np.save(os.path.join(self.path, "paths.npy"), self.paths)
        print(f"db saved to {self.path} successfuly")

    def load(self):
        """Loads the DB"""

        print("loading db...")
        self.index = faiss.read_index(os.path.join(self.path, "embeddings.index"))
        self.paths = np.load(os.path.join(self.path, "paths.npy")).tolist()
        print(f"db loaded from {self.path} successfuly")



    

