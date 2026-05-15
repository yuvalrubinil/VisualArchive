import faiss
import numpy as np
import os
from PIL import Image
from tqdm import tqdm
from img_encoder import ImgEncoder
from text_encoder import TextEncoder
import torch
import gc

VALID_FORMATS = ('.jpg', '.jpeg', '.png')

class VisionDB():
    def __init__(self, device, batch_size=32, hnsw_links=0, load_from=''):
        self.paths = []
        self.index = None
        self.device = device
        self.batch_size = batch_size
        if load_from:
            self.txt_encoder = TextEncoder().to(self.device)
            self.txt_encoder.eval() 
            self.load(load_from)
        else:
            self.img_encoder = ImgEncoder().to(self.device)
            self.img_encoder.eval()
            self.index = faiss.IndexFlatIP(self.img_encoder.dimention) if not hnsw_links else faiss.IndexHNSWFlat(self.img_encoder.dimention, hnsw_links)

    @torch.no_grad()    
    def observe(self, path, to_path):
        assert hasattr(self, 'img_encoder') # in case the DB was init with load_from

        print(f"observing: {path}")
        images = [os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith(VALID_FORMATS)]
        
        # working in batches 
        for i in tqdm(range(0, len(images), self.batch_size)):
            batch_paths = images[i : i + self.batch_size]
            batch_images = []
            valid_batch_paths = []

            # preprocing batch on cpu
            for img_path in batch_paths:
                try:
                    img = Image.open(img_path).convert('RGB')
                    batch_images.append(self.img_encoder.preprocess(img))
                    valid_batch_paths.append(img_path)
                except Exception as e:
                    print(f"skipping {img_path}: {e}")

            # embedding the images
            if batch_images:
                # stack images into a tensor [B, 3, H, W]
                images_tensor = torch.stack(batch_images).to(self.device)
                
                # embedding the images
                embeddings = self.img_encoder(images_tensor)
                embeddings_np = embeddings.cpu().numpy().astype('float32')
                
                self.index.add(embeddings_np)
                self.paths.extend(valid_batch_paths)
        
        self.save(to_path)
        self.clean_vram()
        print(f"done! indexed {self.index.ntotal}/{len(images)} images at {to_path}")

    @torch.no_grad()
    def search(self, text, k=5):
        # tokenizing
        tokens = self.txt_encoder.tokenizer(text)
        tokens = tokens.to(self.device)

        # embedding the text
        query_vec = self.txt_encoder(tokens)

        D, I = self.index.search(query_vec, k)
        relevant_images =  [self.paths[idx] for idx in I[0] if idx != -1]
        scores = D[0]
        return relevant_images, scores

    def save(self, path):
        faiss.write_index(self.index, os.path.join(path, "embeddings.index"))
        np.save(os.path.join(path, "paths.npy"), self.paths)

    def load(self, path):
        self.index = faiss.read_index(os.path.join(path, "embeddings.index"))
        self.paths = np.load(os.path.join(path, "paths.npy")).tolist()

    def clean_vram(self):
        print("purging vision backbone...")

        # delete img_encoder
        if hasattr(self, 'img_encoder'):
            self.img_encoder.to('cpu')
            del self.img_encoder
            
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("-vision backbone deleted")

        # initialize the txt_encoder
        print("initializing text backbone...")
        self.txt_encoder = TextEncoder().to(self.device)
        self.txt_encoder.eval()
        print("-text backbone loaded")


    

